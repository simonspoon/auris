# Post-ASR vocabulary correction

Date: 2026-08-27. Task 950.

**Decision: a lexical correction pass over the six hotword terms, run on
decoded transcript text after ASR, ported from mesa's task-922 vocabulary
correction rather than invented from scratch.**

## Why this exists

`docs/engine.md` picked parakeet with per-word hotword biasing as the engine,
and its "What it does not fix" section already names the gap directly:
`khora` is never produced by any engine or configuration tested in this spike
— whisper or parakeet, prompted or biased. Every parakeet run at every boost
hears "qora"; raising khora's boost alone changes nothing (`spike/RESULTS.md`
§6, "What this does not fix"). Biasing pushes the acoustic model toward a
vocabulary; it cannot conjure a token sequence the model never produces for
that audio. The gap that's left over is exactly what a correction pass over
known term spellings can close cheaply, because the misses are all
near-homophones — "qora", "corvex", "aurus" — not garbage.

## The decision to port mesa, not invent a new algorithm

The correction pass in `spike/harness/vocab_correct.py` is a faithful port of
four things from the sibling project mesa: `soundKey`, `COMMON_ENGLISH`,
`buildVocabulary` and `correctVocabulary` in
`frontend/src/liveRecognition.ts` (mesa task 922). mesa hit the identical
problem first — the Web Speech API mishearing mesa's own product names for
ordinary words that sound like them ("khora" as "chorus", "helios" as
"helius") — and already fought the edge cases that make a phonetic fold safe
to ship. Re-deriving that from first principles would mean re-discovering the
same failure modes mesa already paid for. Consistency with mesa's algorithm
was treated as more valuable than any taste difference in how auris might
have built it independently.

One ordering detail in the fold is load-bearing and was preserved exactly:
**doubled letters collapse to one *before* vowels are stripped, not after.**
Reversing that order breaks the flagship case the whole function exists to
solve. `kokoro` strips vowels first would give `kkr`, which then collapses to
`kr` — exactly `khora`'s key. With both terms landing on the same key,
`build_vocabulary`'s ambiguity rule (a key claimed by two different terms is
dropped rather than guessed at) would cancel *both* keys, and the pass would
lose the ability to correct either "kakoro" or "qora"/"cora"/"korra". Doing
the collapse first keeps a vowel-separated doubled consonant intact long
enough to matter — `kokoro`'s two `k`s only merge into a run after the vowel
between them is gone, by which point `khora`'s single `k` was never doubled
to begin with, so the two keys stay distinct.

## Term source: read from the hotwords file, not a separate list

Terms are read from the same file that feeds the recognizer's hotword
biasing, `spike/fixtures/hotwords.txt` (`read_hotword_terms`), so the
correction vocabulary and the biasing vocabulary cannot drift apart — adding
or removing a project name only ever requires touching one file. Format is
one `term :boost` line per term (blank lines and `#` comments ignored); the
correction pass parses the term and discards the boost, since boost tuning is
a biasing-time concern the correction pass has no use for:

```
mesa :3.0
auris :3.0
khora :6.5
qorvex :5.0
helios :3.0
kokoro :3.0
```

## Rejected: `/usr/share/dict/words` as the protected-word guard

The natural instinct is to guard against over-correction with a real
dictionary rather than a hand-curated list. That was tried and rejected: it
is unusable in both directions on this vocabulary. It contains `cora` — a
real mishearing of `khora` (`spike/RESULTS.md` §3: base.en/small.en
no-prompt both render `khora` as "Cora") that the pass must be free to
correct, so guarding on its presence would silently disable the correction
this pass exists to make. It also *lacks* `cores` — an ordinary English word
one edit away from `khora`'s corrected forms that must never be rewritten —
so it does not even cover the everyday-word side completely. mesa's curated
`COMMON_ENGLISH` set is used instead: a few hundred function words and common
nouns/verbs, deliberately weighted toward the words that sound like mesa's
(and by extension auris's) own names — `course`, `cores`, `corps`, `quote`,
`hello`, `open`, and so on — rather than an attempt at a complete dictionary.

## Term-to-term correction: not attempted, on purpose

None of the six hotword terms will ever be corrected to a *different*
hotword term (`qorvex` is never rewritten to `khora`, etc.), for three
reasons:

1. **No evidence to arbitrate.** By the time the pass runs, all it has is
   decoded text — no acoustic signal, no context. It cannot tell a
   correctly-decoded "qorvex" from a misdecoded "khora" that happened to come
   out sounding like "qorvex." Rewriting exact terms would trade a rare miss
   (an occasional real confusion left uncorrected) for a systematic one
   (breaking every genuine mention of whichever term loses the arbitration).
2. **The failure mode that motivated worrying about this doesn't apply to
   the chosen engine.** The one observed khora→qorvex-shaped confusion in
   this spike came from whisper's initial *prompt* pulling decoded text
   toward *any* prompt term, including into slots where a different one was
   meant (`docs/engine.md`, "Biasing is also better behaved than whisper's
   prompt"). No parakeet run at a usable hotword boost inserted a name that
   was not spoken. The mechanism that would have made term-to-term confusion
   likely isn't present in what auris actually ships.
3. **The algorithm makes it structurally impossible anyway.** The six terms
   fold to six distinct sound keys (asserted in `vocab_correct.py
   --selftest`), so `build_vocabulary` would never even face a choice between
   them today. And if two terms ever *did* collide on a key in the future,
   the ambiguity rule drops that key entirely rather than picking a winner —
   the pass is built to fail closed on an ambiguous case, not guess.

## Capitalization: one deliberate divergence from mesa

mesa substitutes the vocabulary's stored spelling verbatim — its correction
is injecting a proper noun's real spelling into UI/browser text. auris
diverges: it preserves the matched word's own capitalization pattern instead.
ALLCAPS in, ALLCAPS out; Titlecase in, Titlecase out ("Corvex tap failed"
becomes "Qorvex tap failed", not "qorvex tap failed"); anything else gets the
canonical lowercase spelling as stored in the hotwords file. auris is
correcting mid-sentence dictated transcript text, not injecting a name into
chrome around it, so keeping the sentence's own casing pattern reads as
natural continuation rather than an obvious stitch. `score.py` lowercases
both sides before comparing, so this choice cannot move any metric — it only
affects what a person actually reads.

## The second divergence: an edit-distance gate on top of mesa's sound key

mesa's sound-key match, used alone, is not safe at dictionary scale. An
adversarial review built the vocabulary from the real `spike/fixtures/hotwords.txt`
and scanned `/usr/share/dict/web2` (every word of at least 4 letters not
already in `COMMON_ENGLISH`) for anything the sound-key rule would rewrite.
It found **321 ordinary words**, producing real, damaging sentence rewrites:

- "I saw a mouse in the house." → "...a **mesa** in the house."
- "The town hall meeting starts soon." → "...the town **helios** meeting..."
- "That was a curious choice." → "...a **khora** choice."
- "We drove past the quarry." / "Ask her a query about it." → **khora**
- "The Quaker meeting is Sunday." → "The **Kokoro** meeting..."

Also caught: mouse/mice/muse/maze/mousse → mesa; hall/hole/holy/holly/hill/
hull/heel/hail → helios; cross/crass/cure/curry/cherry/cheer/chair/choir →
khora. This is not a porting bug — the fold behaves exactly as mesa built it
to. It is a scale problem: auris's terms are short and vowel-light, so their
sound keys come out at only 2 characters (`mesa`→`ms`, `auris`→`ar`,
`khora`→`kr`, `helios`→`hl`), and a 2-character key is a common prefix of
hundreds of words in the language. mesa's 237-word curated `COMMON_ENGLISH`
was never going to enumerate enough of English to block all of them — it
was tuned against mesa's own vocabulary and mesa's own (much lower-traffic)
browser-dictation surface, not against an open corpus. The task for this
work explicitly requires the pass not rewrite ordinary English near a term;
a passing `"core of"` selftest case proves only that a case someone already
thought to test is safe, not that the algorithm is.

**The fix: require BOTH gates to fire, not just one.** A correction now
requires (1) the sound keys match — mesa's rule, unchanged — **and** (2)
`levenshtein(word.lower(), term) <= 2` on the raw letters, via a small
stdlib edit-distance function added to `vocab_correct.py` (`_levenshtein`;
no third-party dependency, `score.py` was not touched or imported). This is
a pure tightening — it can only reject a correction the sound-key rule would
have made, never approve one that rule would not have — so it cannot
introduce a new miss that wasn't already possible. Verified by hand and by
selftest: every real mishearing in `spike/RESULTS.md` §3 survives (`corvex`
is 1 edit from `qorvex`; `Aorus` is 2 edits from `auris`; `Kakoro` is 1 edit
from `kokoro`), while `hall`/`hole`/`hill` (4+ edits from `helios`),
`cross`/`cherry`/`chair`/`query` (4+ edits from `khora`), and
`mouse`/`muse`/`maze` (3 edits from `mesa`) are all now correctly rejected —
close enough in sound to share a key, nowhere near close enough in spelling
to plausibly be the same word misheard.

**One accepted loss:** mesa's own `chorus` → `khora` fold (same sound key,
edit distance 3) no longer fires for auris. mesa wants that correction for
its own reasons; auris does not need it — `chorus` never appears as a
mishearing in `spike/RESULTS.md` §3 for any engine tested, and it is itself
ordinary English that should not be silently rewritten. Documented and
asserted in `--selftest`.

**Re-scanning after the gate**, the same dictionary produces 47 survivors,
almost all obscure enough that they will not plausibly appear in real
dictation: `aerie`, `aries`, `arius`, `arras`, `arris`, `aurae`, `aureus`,
`aurous` (→auris); `chara`, `chera`, `chora`, `chorai`, `chorea`, `khar`,
`kharia`, `kharua`, `khir`, `kokra`, `kore`, `kori`, `koroa`, `kory` (→khora
or kokoro); `helio`, `heloe` (→helios); `masai`, `massa`, `masu`, `maza`,
`mease`, `meese`, `meissa`, `mese`, `meso`, `messe`, `mesua`, `meuse`,
`mise`, `moosa`, `mose`, `musa` (→mesa). Checked exhaustively (a
case-insensitive whole-word search for all 47 across every file in
`spike/results/raw/` and `spike/results/hotwords/`): exactly one occurs —
`messe`, once, in `spike/results/raw/whisper-medium.en-noprompt.tsv` u02:
*"Hey **Messe** add a note to Cora that headless mode is the default now."*
That `Messe` is itself whisper's own mishearing of `mesa` (the same
utterance mishears `khora` as `Cora` right after it), so the pass correcting
it — `spike/results/corrected/whisper-medium.en-noprompt.tsv` u02: *"Hey
**Mesa** add a note to **Khora** that headless mode is the default now."* —
is a genuine fix, not collateral damage. The other 46 do not appear
anywhere in either directory.

An independent reviewer's own scan of `/usr/share/dict/web2` reproduced the
same count, 47, but not an identical set — a case-sensitive pass surfaces
`Cora`/`cora`/`Kora`/`kora` and other case-duplicates that a
case-insensitive count collapses. The two 47s should be read as
corroborating the same order of magnitude and the same near-total
elimination of the original 321, not as an exact set match.

**Of the words genuinely common enough in ordinary dictation to be worth an
explicit guard**, eight were added to a new, separate constant,
`AURIS_COMMON_ENGLISH` in `vocab_correct.py` — kept apart from mesa's
`COMMON_ENGLISH` so the ported set stays diff-able against mesa's own file:
`muse`, `muses`, `messy`, `aria`, `arias`, `aura`, `masa`, `mesas`. None of
these appear in any raw or hotwords transcript either, so adding them costs
nothing measured here.

**Three words were deliberately left OUT of that addendum despite passing
both gates, and this is a genuine, unresolved tension, reported rather than
hidden:** `cora`, `kora`, and `auras` are each *also* a literal ASR
misdecode this pass must keep fixing. `spike/results/raw/whisper-medium.en-noprompt.tsv`
u07 literally reads "blocked on **Kora**"; `spike/RESULTS.md` §3 quotes
"Cora" (base.en no-prompt) and "auras" (medium.en no-prompt, "under
**auras**") as real, observed mishearings of `khora` and `auris`
respectively, and the selftest already requires `auras`→`auris` and
`Cora`/`Quora`/`Korra`→`khora` to keep working. But each of these three
strings is *also* an ordinary word or name in its own right — "Cora" is a
common person's name, "kora" is a real (if less common) word for a stringed
instrument, "auras" is the plural of "aura." Guarding them in
`AURIS_COMMON_ENGLISH` would silence a correction the whole feature exists
to make; leaving them out means the pass will, rarely, rewrite a genuine
"Cora" (someone's name) or "auras" (the plural) in ordinary speech. That
trade was made deliberately, in favor of the correction, not discovered by
accident — there is no gate that resolves it, because the two things really
are indistinguishable from decoded text alone.

## Results

Full before/after table, all 20 raw + hotwords transcripts, via
`spike/harness/run_correction.sh` (reproduction: same command, output under
`spike/results/corrected/`):

| Config | WER before | WER after | ΔWER | F1 before | F1 after | ΔF1 |
|---|---|---|---|---|---|---|
| parakeet-tdt-0.6b-int8 | 5.3% | 1.3% | −4.0p | 66.7% | 96.3% | +29.6p |
| whisper-base.en-noprompt | 9.3% | 6.0% | −3.3p | 44.4% | 78.3% | +33.8p |
| whisper-base.en-prompt | 4.7% | 3.3% | −1.3p | 75.0% | 84.6% | +9.6p |
| whisper-medium.en-noprompt | 6.7% | 1.3% | −5.3p | 52.6% | 96.3% | +43.7p |
| whisper-medium.en-prompt | 3.3% | 1.3% | −2.0p | 83.3% | 96.3% | +13.0p |
| whisper-small.en-noprompt | 7.3% | 4.7% | −2.7p | 60.0% | 83.3% | +23.3p |
| whisper-small.en-prompt | 3.3% | 1.3% | −2.0p | 83.3% | 96.3% | +13.0p |
| whisper-tiny.en-noprompt | 12.0% | 8.0% | −4.0p | 35.3% | 78.3% | +43.0p |
| whisper-tiny.en-prompt | 13.3% | 10.0% | −3.3p | 69.6% | 92.9% | +23.3p |
| hot_bpe_1.0 | 4.0% | 1.3% | −2.7p | 78.3% | 96.3% | +18.0p |
| hot_bpe_2.0 | 4.0% | 1.3% | −2.7p | 78.3% | 96.3% | +18.0p |
| hot_bpe_3.0 | 4.0% | 1.3% | −2.7p | 78.3% | 96.3% | +18.0p |
| hot_bpe_3.5 | 4.0% | 1.3% | −2.7p | 78.3% | 96.3% | +18.0p |
| hot_bpe_4.0 | 6.0% | 4.0% | −2.0p | 80.0% | 92.9% | +12.9p |
| hot_bpe_4.5 | 9.3% | 8.0% | −1.3p | 80.0% | 85.7% | +5.7p |
| hot_bpe_5.0 | 9.3% | 8.7% | −0.7p | 84.6% | 85.7% | +1.1p |
| hot_bpe_6.0 | 22.0% | 20.7% | −1.3p | 71.0% | 78.8% | +7.8p |
| hot_bpe_7.0 | 69.3% | 69.3% | +0.0p | 42.9% | 41.9% | −1.0p |
| mbs | 5.3% | 1.3% | −4.0p | 66.7% | 96.3% | +29.6p |
| **tuned** (parakeet + per-word hotwords — the chosen config) | **4.0%** | **2.0%** | **−2.0p** | **83.3%** | **96.3%** | **+13.0p** |

WER never regresses in any config — it improves or ties everywhere. Name F1
improves in every config except one, `hot_bpe_7.0` (see below).

## What this does not fix

Four specific misses, each traced to a real cause rather than left
unexplained:

- **`Oris` → `auris` is not corrected.** `Oris`'s sound key is `or`;
  `auris`'s is `ar`. The fold deliberately keeps the first letter verbatim
  (soundex's one idea worth keeping) rather than folding it away, on the
  theory that two words starting on genuinely different sounds should stay
  apart. `Oris` and `auris` differ on that very first letter, so this is the
  algorithm working as designed, not a bug — a rewrite here would be exactly
  the kind of "correct" a word to something that wasn't said that this
  feature is built to avoid.
- **`Corex` → `qorvex` is not corrected** (`spike/RESULTS.md` §3, base.en
  no-prompt: "wire up Corex"). `Corex` folds to `krk`; `qorvex` folds to
  `krvk`. The middle consonant that `qorvex`'s "v" contributes has no
  counterpart in "Corex," so the keys diverge. Same category as `Oris`: a
  correct refusal, not a defect.
- **`khora` heard as "core of"** (`spike/RESULTS.md` §3, base.en/small.en
  no-prompt) is not corrected. This is a mishearing of one word as *two*
  words. The pass operates token-wise, matching whole `[A-Za-z']+` spans —
  same as mesa's — so a name split across a word boundary is structurally
  out of scope for a pass built this way, not a missed edge case within it.
- **u06's doubled "Helios" is not, and cannot be, corrected.** The reference
  text contains a deliberate pun — "transcribed helios **as hell EOS**
  again" — and every engine benchmarked hears "hell EOS" as a second,
  correctly-spelled "Helios" (`spike/RESULTS.md` §2 already flagged this:
  helios was never actually a clean control case). A lexical pass only has
  one lever — nudging a *misspelling* toward a known term's spelling — and
  the second "Helios" is not misspelled. Removing a token that is already
  the right spelling of a real term, because it's semantically the wrong
  occurrence, requires understanding the sentence, not just recognizing a
  sound-alike. This is out of scope for what a correction pass over a term
  list can be.

**A garbage transcript can still come out of the pass with more name errors
than it went in with, even with the edit-distance gate in place.**
`hot_bpe_7.0` (RESULTS.md §6's "babble" config, where a hotword boost of 7.0
overwhelms the acoustic model) is the one case where name F1 *drops* after
correction, 42.9% → 41.9%. Before the edit-distance gate this pass made four
changes to that transcript and dropped F1 further, to 40.0%; the gate
correctly rejects two of the four — `qoris-`→`khora-` (edit distance 3) and
`qorus`→`khora` (edit distance 3) no longer fire, exactly as intended, since
neither fragment is close enough to `khora`'s spelling to be a plausible
mishearing of it. One change is scoring-invisible: `helios'`→now correctly
left alone, since stripping the leading/trailing-apostrophe fix (see below)
means the pass recognizes "helios'" is already the term, correctly spelled,
and makes no change — `normalize()` in `score.py` strips the apostrophe for
scoring either way, so this was never visible in the numbers, before or
after. The one change that remains and still costs a point of precision is
u04 `heli`→`helios`: genuinely 2 edits from `helios`, inside the gate, and
the algorithm is doing exactly what it is designed to do — but the babble
transcript already contains one correct "Helios" earlier in the same
utterance, so a second, corrected occurrence lands as a false positive
against a reference with only one real "helios." **This remains a genuine
limitation, not a rounding artifact**, just a smaller one than before the
edit-distance gate: a lexical correction pass can still turn an honest
fragment into a confident, wrong, in-gate correction once the underlying
decode has already failed badly enough to produce that fragment in a slot
where no name was spoken. It is not a reason to avoid the pass — every
config actually intended for production use (anything at a sane hotword
boost, or any whisper prompt config) improves, and the edit-distance gate
closed most of this one config's regression (from −2.9p to −1.0p) — but it
means the pass is not a substitute for keeping the decode itself out of its
own failure mode.

## The acceptance criterion, honestly

The task that motivated this work asked for 100% name F1 on the 8 fixtures
after correction. **That is not attainable, and should not be presented as
close enough to round up.** The maximum name F1 reached by any of the 20
configs tested is 96.3% (13/13 recall, 13/14 precision), reached by 10
configs including `tuned`, the chosen production config. The one residual
error, in every config that reaches 96.3%, is the same one: u06's second
"Helios," described above. Every genuine *miss* — a real occurrence of a
mesa name that the pass failed to recover — has been closed to zero in the
`tuned` config; the entire remaining error budget is one false positive that
is, definitionally, not a spelling problem. No purely lexical correction
pass, mesa's or otherwise, can close that last gap without adding semantic
understanding of the sentence it's operating on — a materially different
(and heavier) kind of system than what this task asked for. If a clean 100%
is ever genuinely required, the honest fix is to re-record u06 without the
pun, not to extend the correction algorithm to chase a fixture artifact.

## Two smaller defects fixed by the same review

- `_read_tsv` used to raise `ValueError` and crash the whole CLI run on a
  single malformed input line (one with no tab). It now skips that line with
  a warning to stderr and returns every well-formed row around it — one bad
  line in a large batch should not lose the rest of the batch.
- A bare trailing apostrophe used to be silently dropped from the output:
  `correct_text("Corvex' settings")` produced `"Qorvex settings"`, losing
  the `'`. The apostrophe-splitting logic used for possessives (`'s`) now
  also splits off a bare trailing `'` and reattaches it after correction, so
  it survives. As a bonus fix in the same code path, a **leading**
  apostrophe (`"'Korra"`) no longer defeats Titlecase detection — it used to
  fall through to the bare lowercase canonical spelling, silently dropping
  the apostrophe *and* the capitalization; it is now split off before the
  case check and reattached after, so `"'Korra"` → `"'Khora"`.

## Where this lives

- `spike/harness/vocab_correct.py` — the pass (module + CLI), including
  `--selftest`.
- `spike/harness/run_correction.sh` — regenerates the table above from
  `spike/results/raw/` and `spike/results/hotwords/`.
- `spike/results/corrected/` — corrected transcripts and scores, one pair per
  config.
- `spike/RESULTS.md` §7 — the same table in the benchmark's own results
  document.
