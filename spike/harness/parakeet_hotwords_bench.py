import os, sys, time, json
import sherpa_onnx, soundfile as sf
MD="/Users/simonspoon/inaros/projects/tools/auris/spike/models/parakeet"
WAV="/Users/simonspoon/inaros/projects/tools/auris/spike/fixtures/wav"
def build(method, hotwords=None, score=2.0, unit="bpe", vocab=None):
    kw=dict(encoder=f"{MD}/encoder.int8.onnx",decoder=f"{MD}/decoder.int8.onnx",
        joiner=f"{MD}/joiner.int8.onnx",tokens=f"{MD}/tokens.txt",num_threads=8,
        sample_rate=16000,feature_dim=80,decoding_method=method,model_type="nemo_transducer")
    if hotwords:
        kw.update(hotwords_file=hotwords,hotwords_score=score,modeling_unit=unit)
        if vocab: kw.update(bpe_vocab=vocab)
    return sherpa_onnx.OfflineRecognizer.from_transducer(**kw)
def run(tag, rec):
    ids=sorted(os.listdir(WAV)); t0=time.time(); out=[]
    for f in ids:
        s,sr=sf.read(f"{WAV}/{f}",dtype="float32")
        st=rec.create_stream(); st.accept_waveform(sr,s); rec.decode_stream(st)
        out.append((f[:-4], st.result.text))
    dt=time.time()-t0
    with open(f"{os.environ['CLAUDE_JOB_DIR']}/tmp/{tag}.tsv","w") as fh:
        for i,t in out: fh.write(f"{i}\t{t}\n")
    print(tag,"decode_s=%.2f"%dt,"rtf=%.3f"%(dt/45.43), file=sys.stderr)
mode=sys.argv[1]
if mode=="mbs": run("mbs", build("modified_beam_search"))
else:
    hw=sys.argv[2]; sc=float(sys.argv[3]); unit=sys.argv[4]; vocab=sys.argv[5] if len(sys.argv)>5 else None
    run(f"hot_{unit}_{sc}", build("modified_beam_search",hw,sc,unit,vocab))
