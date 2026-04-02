export default function ChartCard({ title, eyebrow, actions, children, className = "" }) {
  return (
    <section className={`panel ${className}`.trim()}>
      <div className="panel-top">
        <div>
          {eyebrow ? <div className="panel-eyebrow">{eyebrow}</div> : null}
          <h2>{title}</h2>
        </div>
        {actions ? <div className="panel-actions">{actions}</div> : null}
      </div>
      {children}
    </section>
  );
}
