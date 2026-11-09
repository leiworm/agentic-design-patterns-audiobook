#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
有声教材语音重建工具

解决「机械感」的三层处理：
  1. 换用神经网络中文音色（edge-tts）
  2. 讲课式韵律：按句分段合成、句间/段间插入不等长停顿、段落级语速微调
  3. 音质：去浑浊 EQ + 清晰度提升 + 轻压缩 + 响度归一化 + 淡入淡出

用法：
  pip install edge-tts                       # 需要联网
  python3 tools/tts_build.py --list-voices
  python3 tools/tts_build.py --sample        # 生成 声音试听/ 下的对比样音
  python3 tools/tts_build.py --all --voice zh-CN-YunxiNeural --apply

--apply 会覆盖 audio/*.mp3，并同步 lessons.json / data.js 的 duration 与逐段
时间轴（页面讲稿高亮依赖）。不加 --apply 只写到 声音试听/重建暂存/。
默认不再生成「全本合辑_33讲.mp3」：它与 33 个分讲内容完全重复、页面也不再引用；
确有整本单文件需求时加 --with-full-book。
"""

import argparse
import array
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FF = shutil.which("ffmpeg") or "/opt/local/bin/ffmpeg"
SAMPLE_RATE = 24000

VOICES = {
    "zh-CN-YunxiNeural": "年轻男声，自然、有亲和力",
    "zh-CN-YunjianNeural": "沉稳男声，纪录片解说感",
    "zh-CN-XiaoxiaoNeural": "温暖女声，播客/有声书感",
    "zh-CN-YunyangNeural": "专业男声，新闻播报感",
    "zh-CN-XiaoyiNeural": "轻快女声，偏活泼",
}

# 各音色默认语速差异很大（实测净语速 320–410 字/分）。
# 按音色校准到约 275 字/分，与原版 279 字/分相当，避免「换音色反而变赶」。
VOICE_RATE = {
    "zh-CN-YunxiNeural": -33,
    "zh-CN-YunjianNeural": -24,
    "zh-CN-XiaoxiaoNeural": -17,
    "zh-CN-YunyangNeural": -26,
    "zh-CN-XiaoyiNeural": -30,
}

# 后期处理链：目标 -16 LUFS（有声书常用），去 240Hz 浑浊、提 3.1kHz 清晰度
POST = (
    "highpass=f=75,"
    "equalizer=f=240:t=q:w=1.0:g=-2.0,"
    "equalizer=f=3100:t=q:w=1.1:g=2.5,"
    "equalizer=f=7600:t=q:w=1.6:g=1.2,"
    "acompressor=threshold=-20dB:ratio=2:attack=12:release=180:makeup=1,"
    "loudnorm=I=-16:TP=-1.5:LRA=11"
)


def wav_duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def ffprobe_duration(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], text=True)
    return float(out.strip())


def split_units(text):
    """把讲稿切成合成单元：(文本, 该单元之后的停顿秒数, 段落序号)"""
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    units = []
    for pi, para in enumerate(paras):
        # 先按句末标点切；超长句再在逗号处切，避免单次请求过长
        sentences = [s.strip() for s in
                     re.findall(r"[^。！？；]*[。！？；]|[^。！？；]+$", para)
                     if s.strip()]
        chunks = []
        for s in sentences:
            if len(s) > 72:
                buf = ""
                for piece in re.findall(r"[^，,]*[，,]|[^，,]+$", s):
                    if buf and len(buf) + len(piece) > 58:
                        chunks.append(buf.strip())
                        buf = piece
                    else:
                        buf += piece
                if buf.strip():
                    chunks.append(buf.strip())
            else:
                chunks.append(s)
        for ci, c in enumerate(chunks):
            last_in_para = ci == len(chunks) - 1
            if last_in_para:
                pause = 0.62                       # 段落之间的「呼吸」
            elif c[-1] in "，,":
                pause = 0.16                       # 长句内部换气
            elif c[-1] == "；":
                pause = 0.34
            else:
                pause = 0.26                       # 句间
            # 强调前的额外留白
            if re.match(r"^(注意|关键|特别|所以|但|这里有个|说到底)", c):
                pause += 0.10
            units.append((c, round(pause, 2), pi))
    return units, paras


def base_rate(voice):
    return VOICE_RATE.get(voice, -28)


def rate_for(pi, para, voice):
    """在音色基准上做小幅段落级变化：开场略快，举例/场景放慢，结论放缓"""
    b = base_rate(voice)
    if re.search(r"比如|假设|例如|举个|典型|想象|场景", para):
        b -= 4
    if pi == 0:
        b += 3
    if re.search(r"最后|小结|记住|总结一下", para):
        b -= 2
    return "%+d%%" % b


def write_silence(path, seconds):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(b"\x00\x00" * int(SAMPLE_RATE * seconds))


def trim_silence(path, thresh_db=-45, pad=0.035):
    """TTS 片段首尾自带约 0.3–0.5s 静音；裁掉后才能精确控制停顿。
    留 35ms 保护带，避免削掉轻声辅音。"""
    with wave.open(path, "rb") as w:
        data = w.readframes(w.getnframes())
    a = array.array("h")
    a.frombytes(data)
    if sys.byteorder != "little":
        a.byteswap()
    limit = int(32768 * 10 ** (thresh_db / 20.0))
    pad_n = int(SAMPLE_RATE * pad)
    first = next((i for i, v in enumerate(a) if abs(v) > limit), None)
    if first is None:
        return
    last = len(a) - 1 - next(
        (i for i, v in enumerate(reversed(a)) if abs(v) > limit), 0)
    start, end = max(0, first - pad_n), min(len(a), last + pad_n + 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(a[start:end].tobytes())


async def synth_units(units, paras, voice, workdir, concurrency=4):
    """并发合成每个单元；失败自动退避重试"""
    import edge_tts
    sem = asyncio.Semaphore(concurrency)

    async def one(i, text, rate):
        dest = os.path.join(workdir, "s%03d.mp3" % i)
        async with sem:
            err = None
            for attempt in range(4):
                try:
                    com = edge_tts.Communicate(text, voice, rate=rate)
                    await com.save(dest)
                    if os.path.getsize(dest) > 1000:
                        return
                    raise RuntimeError("empty audio")
                except Exception as e:
                    err = e
                    await asyncio.sleep(1.2 * (attempt + 1))
            raise RuntimeError("unit %d failed: %s" % (i, err))

    tasks = [one(i, t, rate_for(pi, paras[pi], voice))
             for i, (t, _, pi) in enumerate(units)]
    await asyncio.gather(*tasks)


def render(units, workdir, out_path, fade=True):
    """片段 -> wav -> 插入停顿 -> concat -> 后期处理 -> mp3"""
    n = len(units)
    wavs = []
    for i in range(n):
        mp3 = os.path.join(workdir, "s%03d.mp3" % i)
        wav = os.path.join(workdir, "s%03d.wav" % i)
        subprocess.run([FF, "-y", "-v", "error", "-i", mp3,
                        "-ar", str(SAMPLE_RATE), "-ac", "1",
                        "-c:a", "pcm_s16le", wav], check=True)
        trim_silence(wav)
        wavs.append(wav)

    list_path = os.path.join(workdir, "concat.txt")
    para_starts, acc = {}, 0.0
    with open(list_path, "w", encoding="utf-8") as f:
        for i, (_, pause, pi) in enumerate(units):
            if pi not in para_starts:
                para_starts[pi] = round(acc, 2)     # 精确到秒的段落起点
            f.write("file '%s'\n" % wavs[i].replace("'", "'\\''"))
            acc += wav_duration(wavs[i])
            if pause > 0 and i < n - 1:
                sil = os.path.join(workdir, "p%03d.wav" % i)
                write_silence(sil, pause)
                f.write("file '%s'\n" % sil.replace("'", "'\\''"))
                acc += pause
    timeline = [para_starts[k] for k in sorted(para_starts)]

    chain = POST
    if fade:
        chain += ",afade=t=in:st=0:d=0.35"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    subprocess.run(
        [FF, "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", list_path, "-af", chain, "-ar", str(SAMPLE_RATE),
         "-ac", "1", "-c:a", "libmp3lame", "-b:a", "96k", out_path],
        check=True)
    return ffprobe_duration(out_path), timeline


async def build(text, voice, out_path, prosody=True, post=True):
    global POST
    saved_post = POST
    if not post:
        POST = "anull"
    try:
        with tempfile.TemporaryDirectory(prefix="adp_tts_") as wd:
            if prosody:
                units, paras = split_units(text)
                await synth_units(units, paras, voice, wd)
                dur, timeline = render(units, wd, out_path, fade=post)
            else:
                import edge_tts
                raw = os.path.join(wd, "raw.mp3")
                flat = re.sub(r"\n+", "", text)
                await edge_tts.Communicate(
                    flat, voice, rate="%+d%%" % base_rate(voice)).save(raw)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                chain = POST + ",afade=t=in:st=0:d=0.35" if post else "anull"
                subprocess.run([FF, "-y", "-v", "error", "-i", raw,
                                "-af", chain, "-ar", str(SAMPLE_RATE),
                                "-c:a", "libmp3lame", "-b:a", "96k", out_path],
                               check=True)
                dur = ffprobe_duration(out_path)
                timeline = []
            return dur, timeline
    finally:
        POST = saved_post


def load_lessons():
    return json.load(open(os.path.join(ROOT, "lessons.json"), encoding="utf-8"))


def save_durations(durations, timelines):
    """同步 lessons.json 与 data.js 的 duration + 精确段落时间轴（页面高亮依赖）"""
    jp = os.path.join(ROOT, "lessons.json")
    lessons = json.load(open(jp, encoding="utf-8"))
    for i, d in enumerate(durations):
        if d:
            lessons[i]["duration"] = int(round(d))
        tl = timelines[i] if i < len(timelines) else None
        if tl and len(tl) == len(
                [p for p in re.split(r"\n+", lessons[i]["script"]) if p.strip()]):
            lessons[i]["timeline"] = tl
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(lessons, f, ensure_ascii=False, indent=2)
    with open(os.path.join(ROOT, "data.js"), "w", encoding="utf-8") as f:
        f.write("window.LESSONS = " + json.dumps(lessons, ensure_ascii=False,
                                                separators=(",", ":")) + ";\n")


def cmd_sample():
    lessons = load_lessons()
    paras = [p.strip() for p in re.split(r"\n+", lessons[0]["script"]) if p.strip()]
    text = "\n\n".join(paras[:2])       # 与 A 样音截取的前 ~80 秒内容一致
    out = os.path.join(ROOT, "声音试听")
    os.makedirs(out, exist_ok=True)      # 试听目录可能已被清理，需要时自动重建

    src = os.path.join(ROOT, lessons[0]["audio"])
    # A / B：同一段内容的当前音频与仅做音质增强版
    subprocess.run([FF, "-y", "-v", "error", "-ss", "0", "-t", "80",
                    "-i", src, "-ar", str(SAMPLE_RATE),
                    "-c:a", "libmp3lame", "-b:a", "96k",
                    os.path.join(out, "A_当前版.mp3")], check=True)
    subprocess.run([FF, "-y", "-v", "error", "-ss", "0", "-t", "80",
                    "-i", src, "-af", POST +
                    ",afade=t=in:st=0:d=0.35,afade=t=out:st=79:d=0.7",
                    "-ar", str(SAMPLE_RATE), "-c:a", "libmp3lame", "-b:a", "96k",
                    os.path.join(out, "B_音质增强.mp3")], check=True)

    print("A/B 完成，开始合成新音色样音 ...")
    plan = [
        ("C_新音色直读.mp3", "zh-CN-YunxiNeural", False, False),
        ("D_男声自然_全方案.mp3", "zh-CN-YunxiNeural", True, True),
        ("E_男声沉稳_全方案.mp3", "zh-CN-YunjianNeural", True, True),
        ("F_女声温暖_全方案.mp3", "zh-CN-XiaoxiaoNeural", True, True),
    ]
    for name, voice, prosody, post in plan:
        d, _ = asyncio.run(build(text, voice, os.path.join(out, name), prosody, post))
        print("  %-26s %6.1fs  %s" % (name, d, VOICES.get(voice, voice)))


def cmd_all(voice, apply, only, full_book=False):
    lessons = load_lessons()
    n = len(lessons)
    durations = [None] * n
    timelines = [None] * n
    # 先全部合成到暂存区；只有 100% 成功才替换正式音频，避免半新半旧
    stage = os.path.join(ROOT, "声音试听", "重建暂存")
    os.makedirs(stage, exist_ok=True)
    meta_path = os.path.join(stage, "meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    if meta and any(v.get("voice") != voice for v in meta.values()):
        print("检测到音色变更，清空暂存区重新合成", flush=True)
        for fn in os.listdir(stage):
            if fn.endswith(".mp3"):
                os.remove(os.path.join(stage, fn))
        meta = {}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
    idxs = only or list(range(n))

    for pos, i in enumerate(idxs, 1):
        L = lessons[i]
        dest = os.path.join(stage, os.path.basename(L["audio"]))
        key = str(i)
        if key in meta and os.path.exists(dest) and os.path.getsize(dest) > 10000:
            durations[i] = meta[key]["duration"]
            timelines[i] = meta[key]["timeline"]
            print("[%2d/%d] 跳过（已完成）%s" % (pos, len(idxs), L["title"]), flush=True)
            continue
        d, tl = asyncio.run(build(L["script"], voice, dest, True, True))
        durations[i], timelines[i] = d, tl
        meta[key] = {"duration": d, "timeline": tl, "voice": voice}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        print("[%2d/%d] %s  %6.1fs  %d段" % (
            pos, len(idxs), L["title"], d, len(tl)), flush=True)

    if not apply:
        print("预览完成（未改动正式音频）：%s" % stage, flush=True)
        return

    total = os.path.join(ROOT, "全本合辑_33讲.mp3")
    # 只有尚未重建过的原始音频才值得备份（原始 lessons.json 没有 timeline 字段）。
    # 原版备份已按用户要求删除；若仍用「目录是否存在」判断，再次 --apply（如换音色）
    # 会把已合成的新音频复制成一份名为「原版」的假备份，故改按数据状态判断。
    if not any("timeline" in L for L in lessons):
        bak = os.path.join(ROOT, "audio_原版备份")
        if not os.path.exists(bak):
            shutil.copytree(os.path.join(ROOT, "audio"), bak)
            if os.path.exists(total):
                shutil.copy2(total, os.path.join(bak, "全本合辑_33讲_原版.mp3"))
            print("已备份原音频 → %s" % bak, flush=True)
    else:
        print("当前音频已是重建版，跳过原版备份（需留档请先手动复制 audio/）", flush=True)
    for i in idxs:
        shutil.copy2(os.path.join(stage, os.path.basename(lessons[i]["audio"])),
                     os.path.join(ROOT, lessons[i]["audio"]))
    save_durations(durations, timelines)
    if not full_book:
        print("已跳过全本合辑：内容与 33 个分讲重复、页面不再引用；"
              "需要时用 --with-full-book 生成", flush=True)
        return
    concat = os.path.join(ROOT, ".concat.tmp")
    with open(concat, "w", encoding="utf-8") as f:
        for L in lessons:
            p = os.path.join(ROOT, L["audio"]).replace("'", "'\\''")
            f.write("file '%s'\n" % p)
    subprocess.run([FF, "-y", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", concat, "-c", "copy",
                    "-metadata", "title=智能体设计模式·有声精讲（全%d讲）" % n,
                    "-metadata", "artist=WorkBuddy 有声精讲", total], check=True)
    os.remove(concat)
    print("全本合辑已重建：%.1f 分钟" % (ffprobe_duration(total) / 60), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="生成对比样音")
    ap.add_argument("--all", action="store_true", help="重建全部讲次")
    ap.add_argument("--voice", default="zh-CN-YunxiNeural")
    ap.add_argument("--apply", action="store_true", help="覆盖正式音频并同步时长")
    ap.add_argument("--only", help="只处理指定讲次，如 1,2,3")
    ap.add_argument("--with-full-book", action="store_true",
                    help="重建结束后额外生成 全本合辑_33讲.mp3（默认不生成，内容与分讲重复）")
    ap.add_argument("--list-voices", action="store_true")
    a = ap.parse_args()
    if a.list_voices:
        for k, v in VOICES.items():
            print("%-28s %s" % (k, v))
        return
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        sys.exit("缺少 edge-tts，请先执行：pip install edge-tts")
    if a.sample:
        cmd_sample()
    elif a.all:
        only = [int(x) - 1 for x in a.only.split(",")] if a.only else None
        cmd_all(a.voice, a.apply, only, a.with_full_book)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
