// A single chat message bubble. Renders text, an optional inline chart,
// and source-attribution chips for grounded answers.
import ChartMessage from './ChartMessage';

export default function Message({ role, content, sources, chart, streaming }) {
  const isUser = role === 'user';
  return (
    <div className={`msg-row ${isUser ? 'msg-user' : 'msg-bot'}`}>
      <div className="msg-bubble">
        {content ? (
          <div className="msg-text">{content}</div>
        ) : streaming ? (
          <TypingDots />
        ) : null}

        {chart && <ChartMessage chart={chart} />}

        {!isUser && sources && sources.length > 0 && (
          <div className="msg-sources">
            <span className="sources-label">Sources:</span>
            {sources.map((s) => (
              <span key={s} className="source-chip">{s}</span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function TypingDots() {
  return (
    <div className="typing">
      <span></span>
      <span></span>
      <span></span>
    </div>
  );
}
