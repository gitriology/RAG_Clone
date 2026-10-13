import {
  BookOpen,
  CheckCircle2,
  FileSearch,
  Info,
  MessageCircle,
  ShieldCheck,
  X,
} from "lucide-react";

function HelpDialog({ onClose }) {
  return (
    <div
      className="help-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        className="help-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="help-title"
      >
        <header className="help-dialog-header">
          <div className="help-title-mark">
            <Info size={22} />
          </div>
          <div className="help-title-copy">
            <span className="help-eyebrow">RAGar · USER GUIDE</span>
            <h2 id="help-title">Help &amp; information</h2>
            <p>
              A quick guide to asking questions and understanding your results.
            </p>
          </div>
          <button
            className="help-close"
            type="button"
            onClick={onClose}
            aria-label="Close help"
          >
            <X size={19} />
          </button>
        </header>

        <div className="help-dialog-body">
          <section className="help-intro-card">
            <BookOpen size={20} />
            <div>
              <strong>What is RAGar?</strong>
              <p>
                RAGar searches your connected knowledge base for relevant
                evidence, then uses that evidence to compose an answer. It is
                designed to make both the answer and its supporting material
                easier to inspect.
              </p>
            </div>
          </section>

          <h3>How to use RAGar</h3>
          <div className="help-steps">
            <article>
              <span className="help-step-number">01</span>
              <div>
                <strong>Ask a focused question</strong>
                <p>
                  Enter your question in the box at the bottom. Include
                  important names, terms, or constraints. Press Enter to send;
                  use Shift + Enter only if a multi-line input is available.
                </p>
              </div>
            </article>
            <article>
              <span className="help-step-number">02</span>
              <div>
                <strong>Review the answer</strong>
                <p>
                  Read the response and check any cited sources or evidence
                  shown with it. If the answer is incomplete, try a more
                  specific follow-up question.
                </p>
              </div>
            </article>
            <article>
              <span className="help-step-number">03</span>
              <div>
                <strong>Inspect response details</strong>
                <p>
                  Use the Details action on an answer when available to explore
                  retrieval information and evidence summaries. These indicators
                  help explain the pipeline; they are not a guarantee that every
                  statement is correct.
                </p>
              </div>
            </article>
            <article>
              <span className="help-step-number">04</span>
              <div>
                <strong>Return to previous work</strong>
                <p>
                  Open the sidebar to start a New Query or select a recent
                  conversation. Your saved conversations are associated with
                  your signed-in account. You can delete a conversation from its
                  sidebar entry.
                </p>
              </div>
            </article>
          </div>

          <h3>Useful tips</h3>
          <div className="help-tips-grid">
            <div>
              <FileSearch size={18} />
              <strong>Be specific</strong>
              <p>
                Ask one clear question at a time and mention the topic or
                document you mean.
              </p>
            </div>
            <div>
              <MessageCircle size={18} />
              <strong>Refine naturally</strong>
              <p>
                Ask follow-ups to clarify, compare, or narrow the previous
                answer.
              </p>
            </div>
            <div>
              <ShieldCheck size={18} />
              <strong>Verify important claims</strong>
              <p>
                For consequential decisions, check the original source material
                rather than relying only on a generated answer.
              </p>
            </div>
            <div>
              <CheckCircle2 size={18} />
              <strong>Keep your history private</strong>
              <p>
                Sign out when using a shared device. Only access conversations
                from your own account.
              </p>
            </div>
          </div>

          <p className="help-note">
            <strong>Please note:</strong> RAGar can misunderstand a question,
            retrieve incomplete evidence, or generate an inaccurate summary.
            Availability of sources and detail panels depends on the response
            and the connected knowledge base.
          </p>
        </div>
        <footer className="help-dialog-footer">
          <span>RAGar · Search · Reason · Generate</span>
          <button type="button" onClick={onClose}>
            Got it
          </button>
        </footer>
      </section>
    </div>
  );
}

export default HelpDialog;
