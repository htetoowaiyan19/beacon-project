"""Exercise process controls and reporting without loading the 4B model."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.training_ui import TrainingRun, validate_config, DEFAULTS, active_files
from scripts.utils.training_status import StatusWriter, read_status


def config(tmp_path):
    model = tmp_path / 'model'; model.mkdir()
    data = tmp_path / 'data.jsonl'; data.write_text('{}')
    return dict(DEFAULTS, model_path=str(model), train_file=str(data), val_file=str(data), test_file=str(data))


def wait_exit(run):
    deadline = time.time() + 10
    while time.time() < deadline:
        if (run.directory / 'exit.json').exists(): return
        time.sleep(.05)
    raise AssertionError('Child process did not finish')


def test_safe_stop_and_persisted_logs(tmp_path):
    settings = config(tmp_path)
    run = TrainingRun(tmp_path)
    child = "import pathlib,time; print('trainer started',flush=True); root=pathlib.Path('outputs/training_runs');\nwhile not list(root.glob('*/stop.request')): time.sleep(.01)\nprint('saved safely',flush=True)"
    run.start(settings, [sys.executable, '-u', '-c', child])
    with pytest.raises(RuntimeError): run.start(settings)
    assert run.running
    run.stop(); wait_exit(run)
    assert not run.running
    assert 'saved safely' in (run.directory / 'train.log').read_text()
    assert run.stats()['returncode'] == 0
    saved = json.loads((run.directory / 'config.json').read_text())
    assert Path(saved['output_dir']).parent == run.directory


def test_failed_child_is_not_completed(tmp_path):
    run = TrainingRun(tmp_path)
    run.start(config(tmp_path), [sys.executable, '-c', 'raise RuntimeError("simulated OOM")'])
    wait_exit(run)
    assert run.stats()['status'] == 'failed'
    assert 'simulated OOM' in (run.directory / 'train.log').read_text()


def test_invalid_settings_and_active_release(tmp_path):
    settings = config(tmp_path)
    for key, value in [('grad_accum', '0'), ('learning_rate', 'nan'), ('max_steps', '-2')]:
        with pytest.raises(ValueError): validate_config(dict(settings, **{key: value}))
    assert all(Path(p).is_file() for p in active_files().values())


def test_callback_reports_loss_and_stops_at_step(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from train_lora import LiveStatusCallback
    writer = StatusWriter(tmp_path / 'status.json')
    stop = tmp_path / 'stop.request'
    callback = LiveStatusCallback(writer, stop)
    state = SimpleNamespace(global_step=10, max_steps=20, epoch=.5, best_metric=None, best_model_checkpoint=None)
    control = SimpleNamespace(should_training_stop=False, should_save=False)
    callback.on_log(None, state, control, logs={'loss': 1.2, 'epoch': .5, 'learning_rate': .0001})
    assert read_status(writer.path)['loss_step'] == 10
    stop.touch()
    callback.on_step_end(None, state, control)
    assert control.should_training_stop and control.should_save and callback.stopped
    assert read_status(writer.path)['status'] == 'stopping'


def test_real_trainer_honors_stop_and_saves_checkpoint(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from train_lora import LiveStatusCallback, build_training_arguments, training_schedule
    from datasets import Dataset
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    from trl import SFTTrainer
    tokenizer = AutoTokenizer.from_pretrained(str(Path(__file__).resolve().parents[1] / 'models/qwen3-4b'), local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer), hidden_size=32,
        intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16))
    from peft import LoraConfig, get_peft_model
    tiny_config = model.config
    lora = LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj','v_proj'], task_type='CAUSAL_LM')
    model = get_peft_model(model, lora)
    rows = Dataset.from_list([{'messages': [{'role': 'user', 'content': 'Hello'},
        {'role': 'assistant', 'content': 'Hello, friend.'}]}] * 3)
    stop = tmp_path / 'stop.request'; stop.touch()
    live = LiveStatusCallback(StatusWriter(tmp_path / 'status.json'), stop, checkpoint_steps=1)
    schedule = training_schedule(SimpleNamespace(save_steps=1, max_steps=3, batch_size=1, grad_accum=1, epochs=1),len(rows))
    args = build_training_arguments(output_dir=str(tmp_path / 'checkpoints'), use_cpu=True,
        bf16=False, fp16=False, max_steps=3, per_device_train_batch_size=1,
        assistant_only_loss=True, max_length=128, packing=False, loss_type='chunked_nll',
        report_to='none', logging_steps=1, metric_for_best_model='eval_loss', greater_is_better=False,
        **schedule)
    trainer = SFTTrainer(model=model, train_dataset=rows, eval_dataset=rows, args=args,
        processing_class=tokenizer, callbacks=[live])
    trainer.train()
    assert trainer.state.global_step == 1 and live.stopped
    assert (tmp_path / 'checkpoints/checkpoint-1/trainer_state.json').is_file()
    assert read_status(tmp_path / 'status.json')['last_checkpoint'].endswith('checkpoint-1')
    from scripts.utils.training_checkpoints import latest_checkpoint
    checkpoint = latest_checkpoint(tmp_path / 'checkpoints')
    stop.unlink()
    resumed = SFTTrainer(model=get_peft_model(Qwen3ForCausalLM(tiny_config), lora), train_dataset=rows, eval_dataset=rows, args=args,
        processing_class=tokenizer, callbacks=[LiveStatusCallback(StatusWriter(None), checkpoint_steps=1)])
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 3
    assert (tmp_path / 'checkpoints/checkpoint-2/checkpoint_complete.json').is_file()
    assert resumed.state.best_model_checkpoint.endswith('checkpoint-3')


@pytest.mark.parametrize('max_steps,save_steps', [(-1,25),(10,25),(37,25),(3,1)])
def test_production_training_schedule_validates(tmp_path, max_steps, save_steps):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from train_lora import training_schedule, build_training_arguments, LiveStatusCallback
    values = SimpleNamespace(save_steps=save_steps,max_steps=max_steps,batch_size=1,grad_accum=8,epochs=2)
    schedule = training_schedule(values,12759)
    args = build_training_arguments(output_dir=str(tmp_path),use_cpu=True,bf16=False,fp16=False,
        max_steps=max_steps,num_train_epochs=2,per_device_train_batch_size=1,gradient_accumulation_steps=8,
        metric_for_best_model='eval_loss',report_to='none',**schedule)
    assert args.save_steps == args.eval_steps
    assert 'warmup_ratio' not in schedule
    if max_steps == -1:
        assert args.warmup_steps == 96 and args.eval_steps == 250
    callback = LiveStatusCallback(StatusWriter(None),checkpoint_steps=save_steps)
    state = SimpleNamespace(global_step=save_steps,max_steps=3190,epoch=.01,best_metric=None,best_model_checkpoint=None)
    control = SimpleNamespace(should_save=False)
    callback.on_step_end(args,state,control)
    assert control.should_save


def test_status_lock_is_nonfatal_and_recovers(tmp_path, monkeypatch):
    import scripts.utils.training_status as module
    writer = StatusWriter(tmp_path / 'status.json')
    writer.update(step=1)
    replace = module.os.replace
    def locked(*args): raise PermissionError('simulated Windows sharing violation')
    monkeypatch.setattr(module.os, 'replace', locked)
    with pytest.warns(UserWarning, match='training continues'):
        writer.update(step=2)
    assert read_status(writer.path)['step'] == 1
    assert not list(tmp_path.glob('*.tmp'))
    monkeypatch.setattr(module.os, 'replace', replace)
    writer.update(step=3)
    assert read_status(writer.path)['step'] == 3


def test_recovery_skips_incomplete_or_corrupt_checkpoint(tmp_path):
    from scripts.utils.training_checkpoints import seal_checkpoint, latest_checkpoint
    for number in (25,50):
        folder = tmp_path / f'checkpoint-{number}'; folder.mkdir()
        for name in ('trainer_state.json','optimizer.pt','scheduler.pt','rng_state.pth','training_args.bin','adapter_model.safetensors'):
            (folder/name).write_text('test-state')
        seal_checkpoint(folder)
    assert latest_checkpoint(tmp_path).name == 'checkpoint-50'
    (tmp_path/'checkpoint-50/optimizer.pt').write_text('interrupted/corrupt')
    (tmp_path/'checkpoint-75').mkdir()
    assert latest_checkpoint(tmp_path).name == 'checkpoint-25'
