/* Web Worker: エンジンを別スレッドで動かし、UI を止めずに生成と学習を行う */
importScripts('engine.js');
let engine = null;
function handle(msg) {
  const { op } = msg;
  if (op === 'init') {
    engine = new TinyAI.Engine(msg.meta, msg.bin, msg.vocab, msg.kb);
    let restored = false;
    if (msg.snapshot) { try { restored = engine.restore(msg.snapshot); } catch (e) { restored = false; } }
    return { restored, summary: engine.summary(), tokens: restored ? engine.tok.tokens.slice(8) : null };
  }
  if (!engine) throw new Error('not initialised');
  if (op === 'reply') return engine.reply(msg.text, msg.opts);
  if (op === 'learnTurn') return { result: engine.learnTurn(msg.user, msg.bot, msg.context, msg.weight, msg.steps), summary: engine.summary() };
  if (op === 'learnText') return { result: engine.learnText(msg.text, msg.weight), summary: engine.summary() };
  if (op === 'feedback') return { result: engine.feedback(msg.positive), summary: engine.summary() };
  if (op === 'evolveVocab') { const added = engine.evolveVocab(msg.texts); return { added, summary: engine.summary(), tokens: added ? engine.tok.tokens.slice(8) : null }; }
  if (op === 'snapshot') return engine.snapshot();
  if (op === 'summary') return engine.summary();
  throw new Error('unknown op ' + op);
}
self.onmessage = (ev) => {
  const { id } = ev.data;
  try { self.postMessage({ id, ok: true, result: handle(ev.data) }); } catch (e) { self.postMessage({ id, ok: false, error: String(e && e.stack || e) }); }
};
