"""Save narrowly scoped final-scan corrections; preserve raw files and releases."""
import copy
import json
import re
from pathlib import Path
if __package__:
    from .build_seminar_dataset import ROOT, NORMALIZED, RECOVERED, REVIEW_FILE, SYSTEM, content_hash
else:
    from build_seminar_dataset import ROOT, NORMALIZED, RECOVERED, REVIEW_FILE, SYSTEM, content_hash

TLS = 'TLS 1.3 မှာ ပုံမှန် full handshake ကို 1-RTT နဲ့ ပြုလုပ်နိုင်ပြီး static RSA key exchange နဲ့ cipher အဟောင်းတွေကို ဖယ်ရှားထားပါတယ်ဗျ။ Session ပြန်ဆက်ရာမှာ 0-RTT early data ကို ရွေးချယ်သုံးနိုင်ပေမယ့် replay attack အန္တရာယ်ရှိလို့ application ဘက်က ကာကွယ်မှု လိုအပ်ပါတယ်။'
HSTS = 'Browser က HSTS policy ကို လက်ခံမှတ်သားပြီးနောက် သတ်မှတ်ထားတဲ့ max-age ကာလအတွင်း အဲဒီ host ရဲ့ HTTP URL တွေကို HTTPS သို့ ပြောင်းပြီး ချိတ်ဆက်ပေးပါတယ်။ includeSubDomains ပါရင် subdomain တွေကိုလည်း သက်ရောက်ပါတယ်။ Preload မရှိသေးတဲ့ site ကို ပထမဆုံး HTTP နဲ့ ဝင်ချိန်မှာတော့ HSTS header တစ်ခုတည်းက မကာကွယ်နိုင်သေးပါဘူး။'
FACTS = {
    'human-031026data372-00156': ('WPA3-Personal က WPA2-Personal ရဲ့ PSK authentication အစား SAE ကို သုံးပြီး offline password guessing ကို ပိုခံနိုင်ရည်ရှိစေပါတယ်ဗျ။ Password မလိုတဲ့ open Wi-Fi မှာ encryption ပေးတဲ့ OWE က Wi-Fi Enhanced Open နည်းပညာဖြစ်ပြီး WPA3-Personal နဲ့ သီးခြားဖြစ်ပါတယ်။', 'Separate WPA3-Personal/SAE from Enhanced Open/OWE.', 'https://source.android.com/docs/core/connect/wifi-wpa3-owe'),
    'legacy-train-04555': ('HDLC ပုံစံ bit stuffing မှာ 1 ငါးလုံး ဆက်တိုက်တွေ့တိုင်း အဲဒီငါးလုံးနောက်မှာ 0 တစ်လုံး ထည့်ပေးပါတယ်။ လက်ခံဘက်က ထည့်ထားတဲ့ 0 ကို ပြန်ဖယ်ပေးပါတယ်။ ခြောက်လုံးပြည့်မှ စောင့်ပြီး ထည့်တာ မဟုတ်ပါဘူး။', 'Correct the contradictory six/five-ones trigger.', 'https://www.rfc-editor.org/rfc/rfc1662'),
    'human-031026data372-00029': ('လုံလောက်တဲ့ အရွယ်အစားနဲ့ error correction ရှိတဲ့ quantum computer တစ်ခုက Shor\'s algorithm ကို သုံးပြီး RSA ရဲ့ integer factorization နဲ့ ECC ရဲ့ discrete logarithm ပြဿနာတွေကို ဖြေရှင်းနိုင်မယ်ဆိုရင် အဲဒီစနစ်တွေကို ခြိမ်းခြောက်နိုင်ပါတယ်။ လက်ရှိ quantum computer တွေက လက်တွေ့သုံး key တွေကို စက္ကန့်ပိုင်းအတွင်း ဖောက်နိုင်ပြီလို့ မဆိုလိုပါဘူး။', 'Remove unsupported seconds-scale/current-capability guarantee.', 'https://csrc.nist.gov/projects/post-quantum-cryptography/faqs'),
    'human-041026data330-00124': (TLS, 'Replace handshake frequency with latency; explain 0-RTT conditions.', 'https://www.rfc-editor.org/rfc/rfc8446.html'),
    'human-031026data372-00140': (TLS, 'Qualify 0-RTT as optional early data with replay risk.', 'https://www.rfc-editor.org/rfc/rfc8446.html'),
    'legacy-train-03394': (TLS, 'Replace absolute algorithm claim with scoped protocol description.', 'https://www.rfc-editor.org/rfc/rfc8446.html'),
    'legacy-train-04771': (TLS, 'Qualify 0-RTT as optional early data with replay risk.', 'https://www.rfc-editor.org/rfc/rfc8446.html'),
    'human-041026data330-00075': (HSTS, 'Add max-age and first-visit/preload scope.', 'https://www.rfc-editor.org/rfc/rfc6797.html'),
    'human-data 1000(p3)-00098': (HSTS, 'Add max-age and first-visit/preload scope.', 'https://www.rfc-editor.org/rfc/rfc6797.html'),
    'human-041026data330-00053': ('Strict က same-site request တွေမှာပဲ cookie ပို့ခွင့်ပြုပါတယ်။ Lax က same-site request တွေအပြင် safe method (ဥပမာ GET) နဲ့ ပြုလုပ်တဲ့ cross-site top-level navigation တွေမှာလည်း ပို့နိုင်ပါတယ်။ None က same-site နဲ့ cross-site နှစ်မျိုးလုံးမှာ ပို့ခွင့်ပြုပြီး Secure attribute လိုအပ်ပါတယ်။ Browser ရဲ့ အခြား cookie policy တွေကြောင့် ပို့မပို့ ကွာနိုင်ပါတယ်။', 'Clarify same-site versus cross-site scope and Secure requirement.', 'https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Set-Cookie'),
    'legacy-train-05831': ('TCP/IP ကို layer အလိုက် သင်တဲ့အခါ peer layer နှစ်ဖက်က message၊ segment စတဲ့ တူညီတဲ့ protocol data unit အမျိုးအစားကို နားလည်ဖလှယ်နိုင်ရမယ်ဆိုတဲ့ သဘောနဲ့ identical objects principle ကို ရှင်းပြနိုင်ပါတယ်။ ဒါကို network တစ်လျှောက် packet ရဲ့ bit တိုင်း မပြောင်းဘူးလို့ မယူဆသင့်ပါဘူး။\n\nRouter က IPv4 packet ကို forward လုပ်ရင် TTL ကို လျှော့ပြီး header checksum ကို ပြန်တွက်ရပါတယ်။ IPv4 fragmentation ဖြစ်နိုင်သလို outgoing link အတွက် link-layer frame အသစ်လည်း လိုပါတယ်။ ထို့ကြောင့် end-to-end payload အဓိပ္ပာယ်နဲ့ hop တစ်ခုချင်းရဲ့ header/frame ကို ခွဲပြီး နားလည်ဖို့ လိုပါတယ်။', 'Replace bit-identical end-to-end/header guarantee with scoped layer abstraction and router behavior.', 'https://www.rfc-editor.org/rfc/rfc1812'),
}
REJECT = {
    'legacy-train-02627': 'Broken literal translation of an English atom joke; unnatural nonsensical Burmese punchline.',
    'legacy-train-03393': 'Incoherent cow/interview/monkey joke with unrelated closing sentence.',
    'legacy-train-01089': 'Assistant claims it attended a real meeting and spoke for the user without context.',
    'legacy-train-01095': 'Assistant invents real home/weekend plans.',
    'legacy-train-04561': 'Assistant invents a physical item on its table for user pickup.',
    'legacy-train-05322': 'Assistant invents membership and availability in real seminar teams.',
    'legacy-train-01268': 'Unverified current school closure and exam timetable.',
    'legacy-train-04395': 'Unverified current classes and tomorrow\'s tutorial subject.',
}

# User delegated these 16 factual/contextual corrections after reviewing the packet.
JWT = 'ပုံမှန် signed JWT ကို JWS compact format နဲ့ ဖော်ပြတဲ့အခါ Header၊ Payload နဲ့ Signature ဆိုပြီး အပိုင်း ၃ ပိုင်း ရှိပါတယ်ဗျ။ Header မှာ algorithm စတဲ့ metadata၊ Payload မှာ claims တွေ ပါပါတယ်။ Signature အပိုင်းက digital signature သို့မဟုတ် secret key ပါတဲ့ MAC ဖြစ်ပြီး token ကို ပြင်ထားမထားနဲ့ သက်ဆိုင်ရာ key နဲ့ ထုတ်ထားမထား စစ်ဖို့ သုံးပါတယ်။ သာမန် hash တစ်ခုတည်း မဟုတ်ပါဘူး။ Encryption လုပ်ထားတဲ့ JWT ကိုတော့ JWE format နဲ့ ဖော်ပြနိုင်လို့ JWT အားလုံး အပိုင်း ၃ ပိုင်းပဲ ရှိတယ်လို့ မဆိုသင့်ပါဘူး။'
CONTEXT_NOTE = 'Remove unsupported local availability or policy; ask for context instead.'
DELEGATED_FACTS = {
 'human-031026data372-00030': ('CRYSTALS-Kyber ကို အခြေခံထားတဲ့ ML-KEM ကို NIST က FIPS 203 အဖြစ် စံသတ်မှတ်ထားပါတယ်ဗျ။ ဒါက ဆက်သွယ်သူနှစ်ဖက် shared secret တစ်ခု ရရှိအောင် လုပ်ပေးတဲ့ key-encapsulation mechanism ဖြစ်ပြီး လုံလောက်တဲ့ quantum computer နဲ့ တိုက်ခိုက်မှုတွေကိုပါ ခံနိုင်ရည်ရှိဖို့ ရည်ရွယ်ထားပါတယ်။ ရရှိတဲ့ secret ကနေ symmetric key ထုတ်ယူပြီး data ကို encrypt လုပ်ရာမှာ သုံးနိုင်ပါတယ်။', 'Clarify standardized ML-KEM and key encapsulation.', 'https://csrc.nist.gov/pubs/fips/203/final'),
 'human-031026data372-00275': (JWT, 'Correct signature/MAC and scope the three-part format.', 'https://www.rfc-editor.org/rfc/rfc7519.html'),
 'human-041026data330-00229': (JWT, 'Correct signature/MAC and scope the three-part format.', 'https://www.rfc-editor.org/rfc/rfc7519.html'),
 'human-031026data980-00104': ('RAID ဆိုတာ disk အများအပြားကို အစုတစ်ခုအဖြစ် စီမံအသုံးပြုတဲ့ နည်းလမ်းပါ။ RAID level ပေါ်မူတည်ပြီး performance ပိုကောင်းစေခြင်း၊ disk ပျက်စီးမှုကို ခံနိုင်ခြင်း စတဲ့ အကျိုးကျေးဇူးတွေ ရနိုင်ပါတယ်။ RAID 0 မှာတော့ redundancy မရှိပါဘူး။ RAID သုံးထားလည်း မှားဖျက်မိခြင်း၊ malware နဲ့ အခြား data ဆုံးရှုံးမှုတွေကို အကုန် မကာကွယ်နိုင်လို့ သီးခြား backup ထားဖို့ လိုပါတယ်။', 'Qualify RAID protection by level and distinguish backup.', 'https://www.dell.com/support/kbdoc/en-us/000128635/dell-servers-what-are-the-raid-levels-and-their-specifications'),
 'human-041026data330-00139': ('လုံလောက်တဲ့ အရွယ်အစားနဲ့ error correction ရှိတဲ့ quantum computer က Shor\'s algorithm ကို သုံးပြီး RSA ရဲ့ integer factorization နဲ့ ECC ရဲ့ discrete logarithm ပြဿနာတွေကို ဖြေရှင်းနိုင်ရင် ဒီစနစ်တွေကို ခြိမ်းခြောက်နိုင်ပါတယ်ဗျ။ လက်ရှိမှာ လက်တွေ့သုံး RSA/ECC key တွေကို အဲဒီလို ဖောက်နိုင်တဲ့ quantum computer မရှိသေးပါဘူး။ အခု encrypt လုပ်ထားတဲ့ data ကို သိမ်းထားပြီး အနာဂတ်မှာ ပြန်ဖောက်နိုင်တဲ့ အန္တရာယ်ကြောင့် post-quantum cryptography ပြောင်းသုံးဖို့ ကြိုတင်ပြင်ဆင်နေရတာပါ။', 'Remove seconds-scale quantum capability claim.', 'https://www.nist.gov/cybersecurity-and-privacy/what-post-quantum-cryptography'),
 'legacy-train-00623': ('ကွန်ပျူတာက တုံ့ပြန်သေးရင် လုပ်လက်စတွေ save လုပ်ပြီး Restart အရင်စမ်းကြည့်နော်။ ခဏခဏ ဟန်းနေရင် ဘယ် OS သုံးလဲ၊ ဘာလုပ်နေချိန်မှာ ဖြစ်လဲ ပြောပေးပါ။ ဖုန်းကို factory reset ချရင် ဖုန်းထဲက data တွေ ပျက်သွားနိုင်လို့ backup အရင်လုပ်ပြီး အကောင့် password တွေ သိထားဖို့ လိုတယ်။ Android လား iPhone လား၊ ဘယ်မော်ဒယ်လဲ ပြောပေးရင် ကိုက်ညီတဲ့ reset အဆင့်တွေ ပြောပေးမယ်။', 'Add backup/data-loss guidance and request device context.', 'https://support.google.com/android/answer/6088915'),
 'legacy-train-04941': ('လူတိုင်းအတွက် အကောင်းဆုံး Antivirus တစ်ခုတည်းတော့ မရှိပါဘူးနော်။ Windows မှာ ပါတဲ့ Microsoft Defender Antivirus ကို update လုပ်ထားပြီး real-time protection ဖွင့်ထားတာက အခြေခံကာကွယ်ရေးအတွက် သုံးနိုင်တဲ့ ရွေးချယ်မှုတစ်ခုပါ။ လိုအပ်တဲ့ feature၊ စက်ရဲ့ performance နဲ့ အသုံးပြုပုံပေါ်မူတည်ပြီး ရွေးသင့်ပါတယ်။ Antivirus ရှိရုံနဲ့ အန္တရာယ်အားလုံး မကာကွယ်နိုင်လို့ OS update နဲ့ backup လည်း လိုပါတယ်။', 'Replace universal antivirus ranking with scoped advice.', 'https://support.microsoft.com/en-us/windows/stay-protected-with-windows-security-2ae0363d-0ada-c064-8b56-6a39afb6a963'),
 'legacy-train-00406': ('SSD သုံးထားတဲ့ ကွန်ပျူတာက HDD သုံးထားတာထက် ပုံမှန်အားဖြင့် Boot ပိုမြန်ပါတယ်။ ဒါပေမယ့် စောင့်ချိန် လုံးဝမရှိတာတော့ မဟုတ်ပါဘူး။ ကြာချိန်က စက်ပစ္စည်း၊ OS၊ startup program တွေနဲ့ update လုပ်နေမနေ စတာတွေပေါ် မူတည်လို့ တိတိကျကျ အချိန်တစ်ခု သတ်မှတ်ပြောလို့ မရပါဘူး။', 'Remove universal boot-time guarantee.', ''),
 'legacy-train-05561': ('ဘယ်နေရာကနေ ဆူးလေကို သွားမှာလဲ၊ လိုင်းကားနဲ့လား တက္ကစီနဲ့လား? စထွက်မယ့်နေရာနဲ့ စီးမယ့်ယာဉ်ပေါ်မူတည်လို့ ကားခ ကွာနိုင်ပါတယ်။ လက်ရှိနှုန်းထားကိုတော့ သက်ဆိုင်ရာယာဉ်လိုင်း ဒါမှမဟုတ် ယာဉ်မောင်းဆီမှာ အတည်ပြုတာ ပိုသေချာမယ်နော်။', CONTEXT_NOTE, ''),
 'legacy-test-00308': ('ရှည်လို့ နေရခက်နေပြီဆို ညှပ်လိုက်ရင် ပေါ့သွားမှာပဲနော်။ ပုံမှန်ညှပ်နေကျ ဆိုင်ရှိလား၊ ပုံစံအသစ်လေး ပြောင်းချင်တာလား?', CONTEXT_NOTE, ''),
 'legacy-train-02049': ('ဘယ်မြို့နယ်၊ ဘယ်လမ်းနားမှာ ရှိနေတာလဲဗျ? နေရာမသိသေးလို့ အနီးဆုံးဝပ်ရှော့ကို တိတိကျကျ မပြောနိုင်သေးဘူး။ ဘီးပေါက်နေရင် ဆက်မစီးဘဲ လုံခြုံတဲ့နေရာမှာ ရပ်ပြီး အနီးအနားကလူ ဒါမှမဟုတ် မြေပုံမှာ ဆိုင်ကယ်ပြင်ဆိုင် ရှာကြည့်ပါနော်။', CONTEXT_NOTE, ''),
 'legacy-train-02106': ('ဘယ်မြို့နယ်၊ ဘယ်လမ်းနားမှာလဲဗျ? နေရာသိရင် ဆိုင်ရှာဖို့ ကူညီနိုင်မယ်။ ဆိုင်နာမည်တွေ ရှိထားရင်လည်း ဈေးနှုန်း၊ မီနူးနဲ့ review တွေ ကြည့်ပြီး နှိုင်းယှဉ်ပေးလို့ ရပါတယ်။', CONTEXT_NOTE, ''),
 'legacy-train-06061': ('အော်၊ ကြောင်လေး စားနေကျအမျိုးအစားကို မှတ်ပြီး ဝယ်သွားလိုက်နော်။ သွားမယ့်ဆိုင်မှာ ပစ္စည်းရှိမရှိ ကြိုမေးလိုက်ရင် အလကားသွားရတာ သက်သာမယ်။', CONTEXT_NOTE, ''),
 'legacy-validation-00026': ('ဘယ်မြို့နယ်၊ ဘယ်လမ်းနားကို ပို့ပေးရမှာလဲဗျ? Delivery အခမဲ့ရမရက ဆိုင်၊ အကွာအဝေး၊ မှာယူတဲ့ပမာဏနဲ့ promotion ပေါ် မူတည်ပါတယ်။ မမှာခင် ဆိုင် ဒါမှမဟုတ် app ထဲမှာ ပို့ခကို စစ်ကြည့်ပါနော်။', CONTEXT_NOTE, ''),
}
for identity in ('legacy-train-02455', 'legacy-train-03677'):
    DELEGATED_FACTS[identity] = ('ဆိုင်ပေါ်မူတည်ပါတယ်ဗျ။ Reservation လက်ခံတဲ့ဆိုင်ဆိုရင် သွားမယ့်နေ့၊ အချိန်နဲ့ လူဦးရေကို ပြောပြီး ဆိုင်ဖုန်း ဒါမှမဟုတ် booking channel ကနေ ကြိုမေးလို့ ရပါတယ်။ ဘယ်လောက်ကြိုတင်ရမလဲ၊ စရံလိုမလိုကိုတော့ ဆိုင်နဲ့ အတည်ပြုလိုက်နော်။', CONTEXT_NOTE, '')
FACTS.update(DELEGATED_FACTS)


def main():
    candidates = []
    for path in sorted(NORMALIZED.glob('*.jsonl')):
        candidates += [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    for split in ('train', 'validation', 'test'):
        for line in (RECOVERED / split / f'{split}_combined.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            row['messages'] = [{'role': 'system', 'content': SYSTEM}] + row['messages'][1:]
            candidates.append(row)
    reviews = {r['id']: r for r in json.loads(REVIEW_FILE.read_text(encoding='utf-8'))} if REVIEW_FILE.exists() else {}
    evidence = []
    for row in candidates:
        identity = row['id']; messages = copy.deepcopy(row['messages']); notes, sources = [], []
        replacement_tag = None
        question = row['messages'][1]['content']
        if (identity.startswith('human-train1224-') and row['tags'] == 'computer hardware'
                and re.search(r'ကား|Coolant|Thermostat', question) and not re.search(r'ဘာသာစကား|လက်ချောင်း', question)):
            replacement_tag = 'automotive_basics'
            notes.append('Correct automotive cooling metadata previously labeled computer hardware; retain the conversation.')
        for m in messages[1:]:
            original = m['content']; text = original
            # Only the known accidental Burmese ZWNJ / ZWSP artifacts; preserve emoji ZWJ.
            text = re.sub(r'[\u200b\u200c]+\u1031([\u1000-\u1021])([\u103b-\u103e]*)', lambda match: match[1] + match[2] + '\u1031', text)
            text = text.replace('\u200b', '').replace('\u200c', '')
            if text != original: notes.append('Remove accidental zero-width marks and repair associated prebase-vowel order.')
            replacements = {'ဖรม': 'ဖရိမ်', 'คอมพิวเตอร์': 'ကွန်ပျူတာ', 'ပิกဆယ်': 'ပစ်ဆယ်',
                            'စอดထည့်': 'ထည့်', 'ကြားမှ တอบခိုးနားထောင်': 'ကြားဖြတ် ခိုးနားထောင်',
                            'ဘက်ကို偏သွား': 'ဘက်ကို ယိမ်းသွား'}
            for old,new in replacements.items():
                if old in text: text = text.replace(old,new); notes.append(f'Replace accidental foreign text: {old} -> {new}')
            m['content'] = text.strip()
        if identity == 'human-train1224-00199':
            messages[1]['content'] = 'ကားအတွင်းခန်း အပူပေးစနစ် မပူရသည့် အကြောင်းရင်းတစ်ခုမှာ အဘယ်နည်း။'
            notes.append('Replace accidental Thai glyphs in Burmese heating question.')
        if identity == 'human-041026data330-00166':
            messages[1]['content'] = 'DNS Tunneling တိုက်ခိုက်မှုကို SOC မှာ ဘယ်လို ရှာဖွေသိရှိနိုင်သလဲ?'
            notes.append('Correct garbled Burmese question wording; preserve answer.')
        if identity == 'human-041026data330-00330':
            messages[1]['content'] = messages[1]['content'].replace('Forensics မာ', 'Forensics မှာ')
            notes.append('Correct Burmese particle typo.')
        if identity in FACTS:
            messages[2]['content'], note, source = FACTS[identity]
            notes.append(note); sources.append(source)
        if identity in REJECT or messages != row['messages'] or replacement_tag:
            entry = {'id': identity, 'content_sha256': content_hash(row),
                     'status': 'rejected' if identity in REJECT else 'approved', 'reviewer': 'codex_final_scan',
                     'notes': REJECT.get(identity) or ' '.join(dict.fromkeys(notes)), 'evidence_urls': sources}
            if identity not in REJECT: entry['replacement_messages'] = messages
            if replacement_tag: entry['replacement_tags'] = replacement_tag
            if identity in reviews and reviews[identity] != entry:
                existing = reviews[identity]
                if existing.get('reviewer') != 'codex_final_scan' or existing['content_sha256'] != entry['content_sha256']:
                    raise ValueError('Existing project decision differs; inspect before replacing: ' + identity)
            reviews[identity] = entry
            evidence.append({'id': identity, 'action': entry['status'], 'notes': entry['notes'],
                'evidence_urls': sources, 'before': row['messages'][1:],
                'after': None if identity in REJECT else messages[1:],
                'before_tags': row['tags'], 'after_tags': replacement_tag or row['tags']})
    REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_FILE.write_text(json.dumps(list(reviews.values()),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    log = ROOT / 'datasets/reviews/it_seminar_v3.final_scan_changes.json'
    log.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'saved_reviews':len(reviews), 'corrections':sum(e['action']=='approved' for e in evidence),
                      'rejections':sum(e['action']=='rejected' for e in evidence)},indent=2))


if __name__ == '__main__':
    main()
