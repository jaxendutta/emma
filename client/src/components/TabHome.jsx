export default function TabHome() {
  return (
    <>
      <div className="hero">
        <div className="eyebrow">Emergency Medicine &ndash; Mentoring Agent</div>
        <h1>Learn about <em>acute</em><br />emergency conditions<br />with <em>Emma</em>.</h1>
        <p className="hero-sub">
          A knowledge-based conversational agent for medical exam preparation,
          covering 8 acute conditions with symptoms, diagnostics, treatments,
          risk factors, and clinical differentiations.
        </p>
        <div className="cta-hint">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path
              d="M2 10.5C2 11.88 3.12 13 4.5 13H11.5C12.88 13 14 11.88 14 10.5V6.5C14 5.12 12.88 4 11.5 4H4.5C3.12 4 2 5.12 2 6.5V10.5Z"
              stroke="#2d6a4f" strokeWidth="1.3"
            />
            <circle cx="5.5" cy="8.5" r="0.8" fill="#2d6a4f" />
            <circle cx="8"   cy="8.5" r="0.8" fill="#2d6a4f" />
            <circle cx="10.5" cy="8.5" r="0.8" fill="#2d6a4f" />
          </svg>
          Open the chat in the bottom-right corner to get started
        </div>
      </div>

      <div className="section">
        <div className="section-title">Covered conditions</div>
        <p className="section-desc">The agent has structured knowledge on all 8 conditions below.</p>
        <div className="grid">
          <div className="card">
            <div className="card-name">Myocardial Infarction</div>
            <div className="badge-row"><span className="dot"></span> High mortality</div>
          </div>
          <div className="card">
            <div className="card-name">Stroke</div>
            <div className="badge-row"><span className="dot"></span> High mortality</div>
          </div>
          <div className="card">
            <div className="card-name">Pulmonary Embolism</div>
            <div className="badge-row"><span className="dot"></span> High mortality</div>
          </div>
          <div className="card">
            <div className="card-name">Sepsis</div>
            <div className="badge-row"><span className="dot"></span> High mortality</div>
          </div>
          <div className="card">
            <div className="card-name">Anaphylaxis</div>
            <div className="badge-row"><span className="dot"></span> High mortality</div>
          </div>
          <div className="card">
            <div className="card-name">Meningitis</div>
            <div className="badge-row"><span className="dot"></span> High mortality</div>
          </div>
          <div className="card">
            <div className="card-name">Appendicitis</div>
            <div className="badge-row"><span className="dot med"></span> Medium mortality</div>
          </div>
          <div className="card">
            <div className="card-name">Diabetic Ketoacidosis</div>
            <div className="badge-row"><span className="dot med"></span> Medium mortality</div>
          </div>
        </div>
      </div>

      <div className="section">
        <div className="section-title">What you can ask</div>
        <p className="section-desc">The agent understands six types of questions about any of the conditions above.</p>
        <div className="intents">
          <div className="intent-row">
            <span className="intent-num">1</span>
            <span className="intent-name">Symptoms</span>
            <span className="intent-eg">&ldquo;What are the symptoms of a heart attack?&rdquo;</span>
          </div>
          <div className="intent-row">
            <span className="intent-num">2</span>
            <span className="intent-name">Diagnosis</span>
            <span className="intent-eg">&ldquo;How is a pulmonary embolism diagnosed?&rdquo;</span>
          </div>
          <div className="intent-row">
            <span className="intent-num">3</span>
            <span className="intent-name">Treatment</span>
            <span className="intent-eg">&ldquo;How do you treat anaphylaxis?&rdquo;</span>
          </div>
          <div className="intent-row">
            <span className="intent-num">4</span>
            <span className="intent-name">Risk factors</span>
            <span className="intent-eg">&ldquo;Who is most at risk for meningitis?&rdquo;</span>
          </div>
          <div className="intent-row">
            <span className="intent-num">5</span>
            <span className="intent-name">Urgency</span>
            <span className="intent-eg">&ldquo;How quickly does a stroke need to be treated?&rdquo;</span>
          </div>
          <div className="intent-row">
            <span className="intent-num">6</span>
            <span className="intent-name">Differentiation</span>
            <span className="intent-eg">&ldquo;What distinguishes meningitis from a bad headache?&rdquo;</span>
          </div>
        </div>
      </div>
    </>
  )
}
