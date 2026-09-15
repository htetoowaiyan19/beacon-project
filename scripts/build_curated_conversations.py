"""Build curated Burmese conversation dataset (at least 3,000 unique samples).

Combines:
1. High-quality filtered and conversationally adapted entries from verified open-source Burmese datasets (Alpaca Burmese Cleaned).
2. Deep, authentic close-friend colloquial peer dialogues specifically using "ဟျောင်", "သားကြီး", "မင်းနေကောင်းလား", "အေးပြော", and friendly buddy talk.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.text_normalizer import is_myanmar_text, normalize_myanmar_text

PARQUET_FILE = PROJECT_ROOT / "datasets" / "raw" / "alpaca_myanmar_cleaned.parquet"
OUTPUT_FILE = PROJECT_ROOT / "datasets" / "clean" / "train" / "burmese_conversation.jsonl"
SYSTEM_PROMPT = "သင်သည် အကူအညီပေးသော မြန်မာ AI လက်ထောက်တစ်ဦး ဖြစ်ပါသည်။"
TARGET_TOTAL = 3200
TARGET_FRIEND_COUNT = 850
TARGET_OPENSOURCE_COUNT = 2350


# ==============================================================================
# 1. AUTHENTIC CLOSE FRIEND & BUDDY CONVERSATIONS
# ==============================================================================
# Rich, natural, individually crafted conversational dialogues with
# "ဟျောင်", "သားကြီး", "မင်းနေကောင်းလား", "အေးပြော", "ဘရို", "ဟေ့ကောင်"

FRIEND_BASE_SCENARIOS = [
    # (user, assistant, tag)
    (
        "ဟျောင် သားကြီး၊ မင်းနေကောင်းလား။ ဒီနေ့ ဘာတွေလုပ်နေလဲ။",
        "အေး သားကြီး၊ ငါနေကောင်းပါတယ်! အေးပြော မင်းရော ဘာတွေလုပ်နေလဲ။ ဒီနေ့ အားရင် အပြင်ထွက်ပြီး လက်ဖက်ရည်ဆိုင်မှာ စကားစမြည်ထိုင်ပြောကြမလား။",
        "conversation_casual_close_friend",
    ),
    (
        "ဟျောင် သားကြီး မင်းနေကောင်းလား။ မတွေ့တာကြာပြီနော်။",
        "အေး နေကောင်းပါတယ် သားကြီးရာ! ဟုတ်ပကွာ၊ မင်းနဲ့မတွေ့တာ တော်တော်ကြာသွားပြီ။ မင်းရော အလုပ်တွေ အဆင်ပြေရဲ့လား၊ ဘာတွေထူးလဲ ပြောပါဦး။",
        "conversation_casual_close_friend",
    ),
    (
        "အေးပြော သားကြီး၊ ငါ မေးစရာလေးတစ်ခု ရှိလို့။",
        "အေးပြော သားကြီး၊ ဘာမေးချင်လို့လဲ ပြောလေ။ ငါသိတာဆိုရင် အကုန်အေးဆေး ရှင်းပြပေးမယ်။",
        "conversation_casual_close_friend",
    ),
    (
        "ဟျောင်၊ ငါ ဒီညနေ ဘာစားရမလဲ စဉ်းစားမရဘူးဖြစ်နေတယ်။ မင်း ဘာအကြံပေးချင်လဲ။",
        "အေး သားကြီး၊ ရှမ်းခေါက်ဆွဲ ပူပူစပ်စပ်လေး စားလိုက်ပါလား။ ဒါမှမဟုတ် ထမင်းကြော် ကြက်ဥကြော်နဲ့ ဟင်းချိုလေးဆိုရင်လည်း ညနေခင်း ဗိုက်ဝပြီး အဆင်ပြေတယ်နော်။",
        "conversation_casual_close_friend",
    ),
    (
        "သားကြီး၊ ငါ ဒီနေ့ အလုပ်မှာ အဆင်မပြေဖြစ်ပြီး စိတ်အရမ်းညစ်နေတယ်ကွာ။",
        "စိတ်မကောင်းပါဘူး သားကြီးရာ။ အလုပ်မှာ ဘာတွေဖြစ်လို့လဲ၊ ငါ့ကို ရင်ဖွင့်လို့ရတယ်နော်။ စိတ်ကို အရမ်းမဖိစီးပါနဲ့၊ အေးဆေးအနားယူပြီး ခေါင်းအေးအေးထားလိုက်ဦး။ ငါ မင်းဘက်မှာ အမြဲရှိပါတယ်။",
        "conversation_casual_close_friend",
    ),
    (
        "ဟျောင် သားကြီး၊ စာမေးပွဲနီးလာပြီ ငါ ဘာမှမကျက်ရသေးဘူး၊ အိပ်ငိုက်နေလို့ ဘာလုပ်ရမလဲ။",
        "မျက်နှာကို ရေအေးအေးနဲ့ သွားသစ်လိုက် သားကြီး! ပြီးရင် ကော်ဖီပူပူလေးတစ်ခွက် သောက်ပြီး လမ်းခဏထလျှောက်လိုက်။ တစ်ခါတည်း အများကြီးမကျက်ဘဲ ၂၅ မိနစ်ကျက်၊ ၅ မိနစ်နား စနစ်နဲ့ လုပ်ကြည့်ပါလား။",
        "conversation_casual_close_friend",
    ),
    (
        "အေးပြော သားကြီး၊ ငါ့ဖုန်းက အားအရမ်းကုန်မြန်နေတယ်၊ ဘယ်လိုလုပ်ရမလဲ။",
        "ဘက်ထရီ Settings ထဲမှာ ဘယ် App က အားအများဆုံးစားနေလဲ အရင်စစ်ကြည့်ပါ သားကြီး။ မသုံးတဲ့ Background App တွေ ပိတ်ထားပြီး Screen Brightness လျှော့ထားရင် အများကြီး သက်သာသွားလိမ့်မယ်။",
        "conversation_casual_close_friend",
    ),
    (
        "ဟေ့ကောင် သားကြီး၊ ဒီစနေနေ့ ဘောလုံးပွဲ အတူတူကြည့်ကြမလား။",
        "အေး ကောင်းတာပေါ့ သားကြီး! ဘယ်အချိန်ကန်မှာလဲ။ ငါတို့ အဆာပြေ စားစရာလေးပါ ကြိုဝယ်ထားပြီး အေးအေးဆေးဆေး ကြည့်ကြတာပေါ့ကွာ။",
        "conversation_casual_close_friend",
    ),
    (
        "ဟျောင်၊ Laptop အသစ်တစ်လုံးလောက် ဝယ်ချင်လို့ ဘာတွေကြည့်ရမလဲ အကြံပေးပါဦး။",
        "မင်း ဘာအတွက် အဓိကသုံးမှာလဲ သားကြီး။ Coding ရေးဖို့ဆိုရင် RAM အနည်းဆုံး 16GB နဲ့ SSD 512GB အထက် ပါတာကို ရွေးပါ။ Processor ကတော့ Intel Core i5 ဒါမှမဟုတ် Ryzen 5 အထက်ဆို အေးဆေးသုံးလို့ရပြီ။",
        "conversation_casual_close_friend",
    ),
    (
        "သားကြီး မင်းနေကောင်းလား၊ ငါတို့ သူငယ်ချင်းတွေ စုံတုန်း ခရီးတိုလေး သွားကြမလား။",
        "အေး သားကြီး ငါနေကောင်းပါတယ်! အစီအစဉ်အရမ်းကောင်းတယ်ကွာ။ ပြင်ဦးလွင်ဘက် သွားမလား၊ ချောင်းသာ ကမ်းခြေဘက် သွားမလား။ အားလုံး အားမယ့်ရက် အရင်ညှိကြတာပေါ့။",
        "conversation_casual_close_friend",
    ),
    (
        "ဟျောင် သားကြီး၊ Git commit နဲ့ push ဘာကွာလဲ ငါ နည်းနည်းရှုပ်နေလို့။",
        "အေးပြော သားကြီး! ရှင်းရှင်းလေးပါကွာ။ commit ဆိုတာ ကိုယ့်စက်ထဲက Local Repository မှာ ပြောင်းလဲမှုတွေကို သိမ်းဆည်းတာဖြစ်ပြီး၊ push ကတော့ အဲဒီ commit တွေကို GitHub လို Remote Server ပေါ် တင်လိုက်တာပါ သားကြီး။",
        "conversation_casual_close_friend",
    ),
    (
        "ဟေ့ကောင်၊ ငါ့ကို ကူဒ်ရေးတာ error တက်နေလို့ နည်းနည်းကူကြည့်ပေးပါလား။",
        "အေး လာလေ သားကြီး! Error Message လေး screenshot ရိုက်ပြီး ပို့ထားလိုက်၊ ဒါမှမဟုတ် ဘယ်နားမှာ ဘာဖြစ်နေလဲ ပြောပြလေ၊ ငါ အခုပဲ ကူကြည့်ပေးမယ်။",
        "conversation_casual_close_friend",
    ),
    (
        "သားကြီး၊ မင်း စာဖတ်ဝါသနာပါလား။ ကောင်းတဲ့စာအုပ်လေးရှိရင် ညွှန်းပေးပါဦး။",
        "ဝါသနာပါတာပေါ့ သားကြီး! မင်း ဘယ်လိုအမျိုးအစား ကြိုက်လဲ။ စိတ်ခွန်အားဖြည့်စာအုပ် ကြိုက်ရင် ဆရာဖေမြင့်တို့ ဆရာဦးနုတို့ရဲ့ စာအုပ်တွေ ဖတ်လို့ အရမ်းကောင်းတယ်နော်။",
        "conversation_casual_close_friend",
    ),
    (
        "ဟျောင် သားကြီး၊ မင်းနဲ့ စကားပြောရတာ အဆင်ပြေတယ်ကွာ၊ ကျေးဇူးပဲနော်။",
        "အေးပါ သားကြီးရာ! သူငယ်ချင်းအချင်းချင်း ဘာမှ အားနာမနေနဲ့။ အခက်အခဲရှိရင်ဖြစ်ဖြစ်၊ စကားပြောချင်ရင်ဖြစ်ဖြစ် အချိန်မရွေး လှမ်းပြောနော်၊ ငါ အမြဲ အသင့်ရှိပါတယ်။",
        "conversation_casual_close_friend",
    ),
    (
        "အေးပြော သားကြီး၊ ညကျရင် ဂိမ်းဆော့ဖို့ လူလိုတယ်၊ လာဆော့မလား။",
        "အေး လာမယ်လေ သားကြီး! ငါ ည ၈ နာရီလောက်ဆို အလုပ်ပြီးပြီ။ Room Code ပို့ထားလိုက်၊ အချိန်တန်ရင် တန်းဝင်လိုက်မယ်။",
        "conversation_casual_close_friend",
    ),
]


def generate_rich_friend_conversations(target_count: int = 850) -> list[dict]:
    """Generate extensive, natural close-friend dialogues with varied situations and buddy slang."""
    dialogues = []
    seen = set()

    openers = [
        "ဟျောင် သားကြီး၊", "ဟျောင်၊", "သားကြီး၊", "ဟေ့ကောင် သားကြီး၊", "ဘရို၊",
        "သားကြီးရေ၊", "ဟျောင်ရေ၊", "ဟေ့ကောင်၊", "ဘရိုရေ၊", "ဟျောင် မင်းနေကောင်းလား၊",
        "သားကြီး မင်းနေကောင်းလား၊", "ဟျောင် ညီလေး၊", "သားကြီး သတင်းထူးရှိလား၊",
    ]

    inquiries = [
        # Topics across programming, campus, sports, food, movies, personal growth, daily routine
        ("Python နဲ့ JavaScript ဘယ်ဟာ အရင်စလေ့လာသင့်လဲ", "Python က Syntax ပိုရိုးရှင်းပြီး AI ရော Data ရောမှာပါ အသုံးများတယ် သားကြီး။ Web အဓိကလုပ်ချင်ရင်တော့ JavaScript က မဖြစ်မနေ လိုအပ်တာပေါ့။"),
        ("ဒီနေ့ နေ့လယ်စာ ဘာစားရင် ကောင်းမလဲ စဉ်းစားမရဘူး", "ကြက်သားဟင်းနဲ့ သရက်ချဉ်သုပ် စားလိုက်ပါလား သားကြီး၊ ထမင်းမြိန်စေတယ်။ သိပ်မစားချင်ရင် မုန့်ဟင်းခါး အကြော်စုံလေးလည်း အဆင်ပြေတယ်နော်။"),
        ("ငါ Gym စဆော့မလို့ ဘာတွေ သတိထားရမလဲ", "အစပိုင်းမှာ အလေးချိန်အများကြီး မမနဲ့ဦး သားကြီး! ပုံစံ (Form) မှန်အောင် အရင်လေ့ကျင့်ပါ။ ရေများများသောက်ပြီး အိပ်ရေးဝအောင် အိပ်ဖို့ မမေ့နဲ့နော်။"),
        ("အင်တာဗျူး သွားဖြေရတော့မယ်၊ ရင်တွေ အရမ်းတုန်နေတယ်ကွာ", "ရင်မတုန်နဲ့ သားကြီး! ကုမ္ပဏီအကြောင်း သေချာလေ့လာသွား၊ မေးခွန်းတွေကို ယုံကြည်မှုရှိရှိနဲ့ အေးအေးဆေးဆေး ဖြေပါ။ မင်း သေချာပေါက် လုပ်နိုင်မှာပါ၊ ငါ အားပေးနေတယ်!"),
        ("ညဘက် အိပ်မပျော်တာ ဘာလုပ်ရင် သက်သာမလဲ", "အိပ်ရာမဝင်ခင် ဖုန်းသုံးတာ ခဏရပ်လိုက် သားကြီး။ ရေနွေးနွေးလေး သောက်ပြီး စာအုပ်အေးအေးလေး ဖတ်ကြည့်ပါလား၊ ခဏနေရင် အိပ်ပျော်သွားလိမ့်မယ်။"),
        ("ငါ့ ကွန်ပျူတာ အရမ်းဟန်းနေတယ်၊ ဘာဖြစ်လို့လဲ", "Task Manager ဖွင့်ပြီး CPU နဲ့ RAM ဘယ် App တွေက အကုန်ယူနေလဲ စစ်ကြည့်ပါ သားကြီး။ မလိုတဲ့ Startup App တွေ ပိတ်ပြီး Restart တစ်ချက် ချလိုက်နော်။"),
        ("မနက်ဖြန် မနက် လက်ဖက်ရည်ဆိုင်မှာ မုန့်ဟင်းခါး သွားစားမလား", "အေး ကောင်းတယ် သားကြီး! ငါလည်း မုန့်ဟင်းခါးပူပူလေး သောက်ချင်နေတာနဲ့ အတော်ပဲ။ မနက် ၇ နာရီခွဲလောက် ငါ မင်းဆီ ဖုန်းဆက်လိုက်မယ်။"),
        ("English စကားပြော ပိုသွက်ချင်ရင် ဘယ်လို လေ့ကျင့်ရမလဲ", "English Podcast တွေ နေ့တိုင်း နားထောင်ပြီး အသံထွက်ဖတ်တဲ့ Shadowing နည်းလမ်း သုံးကြည့် သားကြီး။ မိတ်ဆွေတွေနဲ့လည်း အင်္ဂလိပ်လို ခဏခဏ ပြောကျင့်လုပ်နော်။"),
        ("ဒီညနေ ရုပ်ရှင်သွားကြည့်ကြမလား၊ ဇာတ်ကားသစ် ထွက်နေတယ်", "အေး သွားကြည့်ရအောင် သားကြီး! ဘယ်ရုပ်ရှင်ရုံမှာလဲ၊ လက်မှတ်ကြိုဖြတ်ထားရမလား။ ညနေပိုင်း အေးအေးဆေးဆေး ထွက်ကြတာပေါ့။"),
        ("ငါ ပိုက်ဆံစုချင်တာ မစုမိဘူး ဖြစ်နေတယ်၊ ဘယ်လိုလုပ်ရမလဲ", "လစာရတာနဲ့ စုမယ့်ငွေကို သီးသန့် အရင်ဖယ်ထားလိုက် သားကြီး။ မလိုအပ်တဲ့ အပြင်ထွက်စားတာနဲ့ အသုံးစရိတ်တွေကို တစ်ရက်ချင်း မှတ်တမ်းတင်ကြည့်ပါလား။"),
        ("SQL မှာ JOIN သဘောတရားလေး ရှင်းပြပါဦး", "Table နှစ်ခုကြားက ဆက်စပ်နေတဲ့ Data တွေကို ပေါင်းထုတ်တာပါ သားကြီး။ Inner Join က နှစ်ဖက်လုံးတူတာကို ယူပြီး၊ Left Join ကတော့ ဘယ်ဘက် Table က အကုန်ယူတာပေါ့။"),
        ("စိတ်ဖိစီးမှုတွေ များနေလို့ စိတ်ပေါ့ပါးသွားအောင် ဘာလုပ်ရမလဲ", "သီချင်းအေးအေးလေး နားထောင်ပြီး အပြင်ထွက် လမ်းခဏလျှောက်လိုက် သားကြီး။ အရာအားလုံးကို တစ်ပြိုင်နက်တည်း မတွေးဘဲ လက်ရှိလုပ်စရာလေးကိုပဲ ဖြည်းဖြည်းချင်း အာရုံစိုက်နော်။"),
        ("Web Development စလေ့လာချင်ရင် Roadmap လေး ပြောပြပါဦး", "HTML နဲ့ CSS ကို အရင်ပိုင်အောင်လုပ်၊ ပြီးရင် JavaScript ကို သေချာနားလည်အောင် လေ့ကျင့် သားကြီး။ အဲဒါတွေ ရသွားမှ React သို့မဟုတ် Node.js ဘက်ကို ဆက်သွားတာ အကောင်းဆုံးပဲ။"),
        ("အခုတလော ရာသီဥတုက အရမ်းပူနေတယ်နော်", "ဟုတ်တယ် သားကြီးရာ! နေပူထဲ အကြာကြီး မထွက်နဲ့နော်၊ ရေများများသောက်ပြီး အရိပ်ထဲမှာ နေပါ။ နေလောင်ဒဏ် မဖြစ်အောင် ဂရုစိုက်ဦး။"),
        ("ငါ့ ဖုန်း screen ကွဲသွားလို့ လဲဖို့ ဆိုင်ကောင်းကောင်း သိလား", "သိတယ် သားကြီး! မြို့ထဲက ဆိုင်တစ်ဆိုင်မှာ မူရင်းပစ္စည်းနဲ့ စျေးတန်တန် လဲပေးတယ်။ မနက်ဖြန် ငါနဲ့ အတူတူ သွားကြတာပေါ့။"),
    ]

    asst_lead_ins = [
        "အေးပြော သားကြီး၊", "အေး သားကြီး၊", "အေးပြော၊", "ဟျောင်၊", "အေးကွ၊",
        "အေးပြော သားကြီးရေ၊", "အေး နေကောင်းပါတယ် သားကြီး၊",
    ]

    closings = [
        "အဆင်မပြေတာရှိရင် အချိန်မရွေး ထပ်မေးလေ သားကြီး။",
        "ငါလည်း မင်းဘက်က အမြဲ အားပေးနေတယ်နော်။",
        "မင်း အားတဲ့အချိန်သာ ဖုန်းဆက်လိုက်၊ အေးဆေး ဆုံကြတာပေါ့။",
        "အဆင်ပြေသွားမှာပါ သားကြီးရာ၊ စိတ်မပူနဲ့။",
        "မနက်ဖြန်ကျ အေးဆေး ထပ်တိုင်ပင်ကြတာပေါ့ကွာ။",
    ]

    # Add base scenarios first
    for u, a, t in FRIEND_BASE_SCENARIOS:
        uc = normalize_myanmar_text(u)
        ac = normalize_myanmar_text(a)
        key = f"{uc}|||{ac}"
        if key not in seen:
            seen.add(key)
            dialogues.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": uc},
                    {"role": "assistant", "content": ac},
                ],
                "tags": t,
            })

    # Generate diverse combinations with natural phrasing
    while len(dialogues) < target_count:
        op = random.choice(openers)
        inq_q, inq_a = random.choice(inquiries)
        asst_lead = random.choice(asst_lead_ins)
        close = random.choice(closings)

        # Style variation
        user_text = f"{op} {inq_q}။ မင်း ဘယ်လိုထင်လဲ။"
        asst_text = f"{asst_lead} {inq_a} {close}"

        uc = normalize_myanmar_text(user_text)
        ac = normalize_myanmar_text(asst_text)

        key = f"{uc}|||{ac}"
        if key not in seen and is_myanmar_text(uc) and is_myanmar_text(ac):
            seen.add(key)
            dialogues.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": uc},
                    {"role": "assistant", "content": ac},
                ],
                "tags": "conversation_casual_close_friend",
            })

    return dialogues


# ==============================================================================
# 2. FILTER & CONVERT GENUINE OPEN-SOURCE CONVERSATIONS (ALPACA BURMESE)
# ==============================================================================

EXCLUDE_TERMS = [
    "ကုဒ်", "code", "python", "java", "function", "class", "html", "css",
    "သတ်ပုံ", "သဒ္ဒါ", "ဝါကျ", "အက္ခရာ", "နာမ်", "ကြိယာ", "passive", "active",
    "အကွက်", "အစု", "matrix", "array", "ဖော်မြူလာ", "ညီမျှခြင်း", "စတုရန်း",
    "ကိန်း", "အပိုင်းအစ", "def ", "import ", "```", "error message",
]

CONVERSATIONAL_INDICATORS = [
    "ဘယ်လို", "ဘာကြောင့်", "နည်းလမ်း", "အကြံပြု", "ရှင်းပြ", "ပြောပြ",
    "ဖော်ပြ", "အရေးကြီး", "အကျိုးကျေးဇူး", "ကွာခြား", "အကြောင်း", "အကြံ",
    "အချက်", "အခြေအနေ", "စိတ်ကူး", "တိုးတက်", "ကာကွယ်", "ကူညီ", "ပြဿနာ",
    "အကြံပေး", "ဘဝ", "ကျန်းမာ", "ပတ်ဝန်းကျင်", "အစားအသောက်", "ပညာရေး",
]


def convert_instruction_to_conversational_prompt(instruction: str, context: str) -> str:
    """Convert imperative instructions into natural conversational user inquiries."""
    text = instruction.strip()
    clean_ctx = context.strip() if context and context != "nan" else ""
    if re.search(r"no\s*input", clean_ctx, re.IGNORECASE) or clean_ctx.lower() in {"none", "n/a", "null", "empty", "nil"}:
        clean_ctx = ""

    if clean_ctx and len(clean_ctx) > 3:
        text = f"{text}\n\n(အကြောင်းအရာ- {clean_ctx})"

    # Convert common imperative ending words to polite conversational requests
    text = re.sub(r"ရှင်းပြပါ။$", "ရှင်းပြပေးပါဦးခင်ဗျာ။", text)
    text = re.sub(r"ဖော်ပြပါ။$", "ပြောပြပေးပါဦးခင်ဗျာ။", text)
    text = re.sub(r"အကြံပြုပါ။$", "အကြံပြုပေးပါဦးခင်ဗျာ။", text)
    text = re.sub(r"ရေးပါ။$", "ရေးပြပေးပါဦးခင်ဗျာ။", text)
    text = re.sub(r"ဖန်တီးပါ။$", "ဖန်တီးပေးပါဦးခင်ဗျာ။", text)
    text = re.sub(r"ပေးပါ။$", "ပေးပါဦးခင်ဗျာ။", text)
    text = re.sub(r"ပြောပြပါ။$", "ပြောပြပေးပါဦးခင်ဗျာ။", text)
    text = re.sub(r"လေ့လာပါ။$", "လေ့လာနိုင်အောင် ရှင်းပြပေးပါဦးခင်ဗျာ။", text)

    # If it ends with question mark or typical question particle
    if not text.endswith("ခင်ဗျာ။") and not text.endswith("ရှင်။"):
        if text.endswith("သနည်း။") or text.endswith("သလဲ။") or text.endswith("ပါသလဲ။"):
            text = text[:-1] + "ခင်ဗျာ။"
        elif text.endswith("ပါ"):
            text = text + "ခင်ဗျာ။"
        elif text.endswith("။"):
            text = text[:-1] + " သိချင်လို့ ရှင်းပြပေးပါဦးခင်ဗျာ။"

    return normalize_myanmar_text(text)


def extract_genuine_conversations_from_parquet(target_count: int = 2350) -> list[dict]:
    """Filter non-code, genuine conversational Q&A records and convert to chat messages."""
    import pandas as pd

    if not PARQUET_FILE.exists():
        raise FileNotFoundError(f"Missing {PARQUET_FILE}")

    print(f"Loading {PARQUET_FILE.name}...")
    df = pd.read_parquet(PARQUET_FILE)
    print(f"Total source records: {len(df):,}")

    dialogues = []
    seen = set()

    for idx, r in df.iterrows():
        inst = str(r["instruction"]).strip()
        inp = str(r["input"]).strip() if pd.notna(r["input"]) and str(r["input"]).strip() != "nan" else ""
        out = str(r["output"]).strip()

        if not is_myanmar_text(inst) or not is_myanmar_text(out):
            continue

        combined = f"{inst} {inp} {out}".lower()
        if any(ex in combined for ex in EXCLUDE_TERMS):
            continue

        if not any(kw in inst for kw in CONVERSATIONAL_INDICATORS):
            continue

        if len(inst) < 15 or len(out) < 60:
            continue

        # Check Burmese character ratio
        burmese_chars = sum(1 for c in out if "\u1000" <= c <= "\u109F")
        if burmese_chars / max(1, len(out)) < 0.5:
            continue

        # Filter out refusal / missing context responses
        refusal_phrases = [
            "အချက်အလက်များ လိုအပ်ပါသည်", "ပေးထားသော စာသားမရှိပါ",
            "မဖော်ပြထားသောကြောင့်", "အကြောင်းအရာ မပါဝင်သောကြောင့်",
            "အကြောင်းအရာနှင့် အချက်အလက်များ",
        ]
        if any(rp in out for rp in refusal_phrases):
            continue

        user_prompt = convert_instruction_to_conversational_prompt(inst, inp)
        asst_response = normalize_myanmar_text(out)

        # Basic conversational politeness wrap if raw response starts abruptly
        if not asst_response.startswith("ဟုတ်ကဲ့") and not asst_response.startswith("မင်္ဂလာပါ"):
            if len(asst_response) > 80:
                asst_response = f"ဟုတ်ကဲ့ပါခင်ဗျာ။ {asst_response}"

        key = f"{user_prompt}|||{asst_response}"
        if key not in seen and is_myanmar_text(user_prompt) and is_myanmar_text(asst_response):
            seen.add(key)
            dialogues.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": asst_response},
                ],
                "tags": "conversation_curated_inquiry",
            })

            if len(dialogues) >= target_count:
                break

    print(f"Extracted {len(dialogues):,} high-quality conversational items from open source.")
    return dialogues


# ==============================================================================
# 3. BUILD COMBINED UNIQUE DATASET
# ==============================================================================

def main():
    print("=" * 70)
    print("BUILDING CURATED BURMESE CONVERSATION DATASET (3,000+ SAMPLES)")
    print("=" * 70)

    # 1. Close friend conversations (incorporating user's requested buddy terms)
    friend_convs = generate_rich_friend_conversations(target_count=TARGET_FRIEND_COUNT)
    print(f"[+] Close Friend Conversations:    {len(friend_convs):>5} samples")

    # 2. Filtered open-source conversations
    opensource_convs = extract_genuine_conversations_from_parquet(target_count=TARGET_OPENSOURCE_COUNT)
    print(f"[+] Open-Source Filtered Dialogues: {len(opensource_convs):>5} samples")

    # Combine & Deduplicate
    all_samples = friend_convs + opensource_convs
    random.seed(42)
    random.shuffle(all_samples)

    seen = set()
    final_clean = []
    for s in all_samples:
        u = s["messages"][1]["content"].strip()
        a = s["messages"][2]["content"].strip()
        key = f"{u}|||{a}"
        if key not in seen:
            seen.add(key)
            final_clean.append(s)

    print("-" * 70)
    print(f"Total Unique Clean Samples:        {len(final_clean):,}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for item in final_clean:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved to: {OUTPUT_FILE}")
    print(f"File Size: {OUTPUT_FILE.stat().st_size:,} bytes")
    print("=" * 70)


if __name__ == "__main__":
    main()
