import { Loader2, Rocket, Sparkles, X } from "lucide-react";
import { IDEATION_STRATEGIES, type IdeationStrategy } from "./runs";

export type RunWizardState = {
  step: "topic" | "configure";
  topic: string;
  suggestions: string[];
  strategy: IdeationStrategy;
  calls: number;
  iters: number;
  out: string;
  status: string;
  busy: boolean;
  suggesting: boolean;
};

export function ChatRunWizard({
  wizard,
  onBack,
  onCallsChange,
  onClose,
  onContinue,
  onItersChange,
  onLaunch,
  onOutChange,
  onPickTopic,
  onResuggestOut,
  onStrategyChange,
  onTopicChange,
}: {
  wizard: RunWizardState;
  onBack: () => void;
  onCallsChange: (value: number) => void;
  onClose: () => void;
  onContinue: () => void;
  onItersChange: (value: number) => void;
  onLaunch: () => void;
  onOutChange: (value: string) => void;
  onPickTopic: (value: string) => void;
  onResuggestOut: () => void;
  onStrategyChange: (value: IdeationStrategy) => void;
  onTopicChange: (value: string) => void;
}) {
  return (
    <div className="run-wizard-card">
      <div className="run-wizard-head">
        <div>
          <strong>
            <Rocket size={14} /> New exploration run
          </strong>
          <span>{wizard.step === "topic" ? "Choose a topic" : "Review settings"}</span>
        </div>
        <button disabled={wizard.busy} onClick={onClose} title="Close this run setup." type="button">
          <X size={13} />
        </button>
      </div>
      {wizard.step === "topic" ? (
        <>
          <div className="run-wizard-suggestions">
            {wizard.suggesting ? <span>Generating topic ideas...</span> : null}
            {wizard.suggestions.map((item) => (
              <button className={item === wizard.topic ? "active" : ""} key={item} onClick={() => onPickTopic(item)} type="button">
                {item}
              </button>
            ))}
          </div>
          <label>
            Topic
            <textarea onChange={(event) => onTopicChange(event.target.value)} rows={2} value={wizard.topic} />
          </label>
          <div className="run-wizard-actions">
            <button disabled={!wizard.topic.trim() || wizard.suggesting} onClick={onContinue} type="button">
              Continue
            </button>
          </div>
        </>
      ) : (
        <>
          <label>
            Topic
            <textarea disabled={wizard.busy} onChange={(event) => onTopicChange(event.target.value)} rows={2} value={wizard.topic} />
          </label>
          <div className="run-wizard-grid">
            <label>
              Strategy
              <select disabled={wizard.busy} onChange={(event) => onStrategyChange(event.target.value as IdeationStrategy)} value={wizard.strategy}>
                {IDEATION_STRATEGIES.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Calls
              <input disabled={wizard.busy} min={1} onChange={(event) => onCallsChange(Number(event.target.value))} type="number" value={wizard.calls} />
            </label>
            <label>
              Iters
              <input disabled={wizard.busy} min={1} onChange={(event) => onItersChange(Number(event.target.value))} type="number" value={wizard.iters} />
            </label>
            <label>
              Output folder
              <div className="run-path-row">
                <input disabled={wizard.busy} onChange={(event) => onOutChange(event.target.value)} value={wizard.out} />
                <button disabled={wizard.busy || wizard.suggesting} onClick={onResuggestOut} title="Suggest a short valid output folder." type="button">
                  {wizard.suggesting ? <Loader2 className="spin" size={13} /> : <Sparkles size={13} />}
                </button>
              </div>
            </label>
          </div>
          {wizard.status ? <div className="status-box">{wizard.status}</div> : null}
          <div className="run-wizard-actions">
            <button disabled={wizard.busy} onClick={onBack} type="button">
              Back
            </button>
            <button disabled={wizard.busy || !wizard.topic.trim() || !wizard.out.trim()} onClick={onLaunch} type="button">
              {wizard.busy ? "Starting..." : "Launch run"}
            </button>
          </div>
        </>
      )}
    </div>
  );
}
