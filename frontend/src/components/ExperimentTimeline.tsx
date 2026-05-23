import React, { useState, useEffect } from 'react';
import { AlertOctagon, CheckCircle2, Circle, Clock, Flame, Shield, ArrowRight, ArrowLeft, ClipboardList } from 'lucide-react';
import { api } from '../services/api';

interface Step {
  step_number: number;
  text: string;
  section_title: string;
  reagents?: string[];
  equipment?: string[];
  temperature?: string;
  timing?: string;
  safety_level: string;
  dependencies?: string[];
}

interface Deviation {
  step_number: number;
  description: string;
  severity: 'minor' | 'moderate' | 'critical';
  timestamp: string;
}

interface ExperimentSession {
  id: string;
  protocol_id: string;
  title: string;
  status: string;
  current_step: number;
  total_steps: number;
  deviations?: Deviation[];
  notes?: Record<number, string>;
  timeline?: any[];
}

interface ExperimentTimelineProps {
  experiment: ExperimentSession;
  protocolSteps: Step[];
  onStepChange: (updatedExperiment: ExperimentSession) => void;
}

export const ExperimentTimeline: React.FC<ExperimentTimelineProps> = ({
  experiment,
  protocolSteps,
  onStepChange
}) => {
  const [updating, setUpdating] = useState(false);
  const [stepNote, setStepNote] = useState('');
  const [localNotes, setLocalNotes] = useState<Record<number, string>>(experiment.notes || {});

  useEffect(() => {
    setLocalNotes(experiment.notes || {});
    setStepNote((experiment.notes || {})[experiment.current_step] || '');
  }, [experiment.current_step, experiment.notes]);

  const handleStepAdvance = async (direction: 'next' | 'prev') => {
    let targetStep = experiment.current_step;
    if (direction === 'next' && experiment.current_step < protocolSteps.length) {
      targetStep += 1;
    } else if (direction === 'prev' && experiment.current_step > 1) {
      targetStep -= 1;
    } else {
      return;
    }

    setUpdating(true);
    try {
      // Save notes if any
      const updatedNotes = { ...localNotes, [experiment.current_step]: stepNote };
      setLocalNotes(updatedNotes);
      
      const response = await api.updateExperimentStep(
        experiment.id,
        targetStep,
        stepNote,
        false // manual progression, not marked as automated deviation
      );
      
      onStepChange(response);
    } catch (err) {
      console.error('Failed to change experiment step', err);
    } finally {
      setUpdating(false);
    }
  };

  const handleSaveNote = async () => {
    setUpdating(true);
    try {
      const response = await api.updateExperimentStep(
        experiment.id,
        experiment.current_step,
        stepNote,
        false
      );
      onStepChange(response);
    } catch (err) {
      console.error('Failed to save step notes', err);
    } finally {
      setUpdating(false);
    }
  };

  const currentActiveStepObj = protocolSteps.find(s => s.step_number === experiment.current_step);

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '1fr 300px',
      gap: '20px',
      height: '100%'
    }}>
      {/* Left Pane: Protocol Steps list */}
      <div className="glass-panel" style={{
        padding: '24px',
        overflowY: 'auto',
        maxHeight: 'calc(100vh - 180px)',
        display: 'flex',
        flexDirection: 'column',
        gap: '24px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <h3 style={{ fontFamily: 'var(--font-title)', fontSize: '18px', fontWeight: 600 }}>
              {experiment.title || 'Experimental Run'}
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '12px', marginTop: '4px' }}>
              Step {experiment.current_step} of {protocolSteps.length}
            </p>
          </div>

          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              className="btn-secondary"
              onClick={() => handleStepAdvance('prev')}
              disabled={experiment.current_step <= 1 || updating}
              style={{ padding: '8px 12px' }}
            >
              <ArrowLeft size={16} />
            </button>
            <button
              className="btn-primary"
              onClick={() => handleStepAdvance('next')}
              disabled={experiment.current_step >= protocolSteps.length || updating}
              style={{ padding: '8px 16px' }}
            >
              Next Step
              <ArrowRight size={16} />
            </button>
          </div>
        </div>

        {/* Steps Timeline Track */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
          paddingLeft: '32px'
        }}>
          {/* Vertical Track Line */}
          <div style={{
            position: 'absolute',
            left: '11px',
            top: '12px',
            bottom: '12px',
            width: '2px',
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
            zIndex: 0
          }} />

          {protocolSteps.map((step) => {
            const isCompleted = step.step_number < experiment.current_step;
            const isActive = step.step_number === experiment.current_step;
            
            let statusIcon = <Circle size={14} style={{ color: 'var(--text-muted)' }} />;
            if (isCompleted) {
              statusIcon = <CheckCircle2 size={16} style={{ color: 'var(--accent)' }} />;
            } else if (isActive) {
              statusIcon = <Circle size={16} style={{ color: 'var(--primary)', fill: 'var(--primary-glow)' }} />;
            }

            return (
              <div
                key={step.step_number}
                className="animate-fade"
                style={{
                  position: 'relative',
                  paddingBottom: '24px',
                  opacity: isActive ? 1 : isCompleted ? 0.6 : 0.4,
                  transition: 'opacity 0.2s ease',
                  zIndex: 1
                }}
              >
                {/* Node indicator */}
                <div style={{
                  position: 'absolute',
                  left: '-32px',
                  top: '2px',
                  width: '24px',
                  height: '24px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: 'var(--bg-surface)',
                  borderRadius: '50%'
                }}>
                  {statusIcon}
                </div>

                {/* Step contents */}
                <div style={{
                  background: isActive ? 'rgba(99, 102, 241, 0.03)' : 'transparent',
                  border: isActive ? '1px solid rgba(99, 102, 241, 0.15)' : '1px solid transparent',
                  borderRadius: 'var(--radius-md)',
                  padding: isActive ? '16px' : '0 16px 0 0',
                }}>
                  <h4 style={{
                    fontWeight: 600,
                    fontSize: '14px',
                    color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                    marginBottom: '4px'
                  }}>
                    Step {step.step_number}: {step.section_title}
                  </h4>
                  <p style={{
                    color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                    fontSize: '13px',
                    lineHeight: 1.4
                  }}>
                    {step.text}
                  </p>

                  {/* Active Step metadata parameters */}
                  {isActive && (
                    <div style={{
                      display: 'flex',
                      flexWrap: 'wrap',
                      gap: '8px',
                      marginTop: '12px'
                    }}>
                      {step.temperature && (
                        <div style={{
                          background: 'rgba(255,255,255,0.04)',
                          border: '1px solid var(--border-color)',
                          padding: '2px 8px',
                          borderRadius: '12px',
                          fontSize: '11px',
                          color: 'var(--text-secondary)',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px'
                        }}>
                          <Flame size={12} style={{ color: 'var(--danger)' }} />
                          {step.temperature}
                        </div>
                      )}
                      {step.timing && (
                        <div style={{
                          background: 'rgba(255,255,255,0.04)',
                          border: '1px solid var(--border-color)',
                          padding: '2px 8px',
                          borderRadius: '12px',
                          fontSize: '11px',
                          color: 'var(--text-secondary)',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px'
                        }}>
                          <Clock size={12} style={{ color: 'var(--primary)' }} />
                          {step.timing}
                        </div>
                      )}
                      {step.safety_level && step.safety_level !== 'low' && (
                        <div style={{
                          background: step.safety_level === 'critical' ? 'var(--danger-glow)' : 'var(--warning-glow)',
                          border: `1px solid ${step.safety_level === 'critical' ? 'var(--danger)' : 'var(--warning)'}`,
                          padding: '2px 8px',
                          borderRadius: '12px',
                          fontSize: '11px',
                          color: step.safety_level === 'critical' ? 'var(--danger)' : 'var(--warning)',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px',
                          fontWeight: 500
                        }}>
                          <Shield size={12} />
                          Safety: {step.safety_level.toUpperCase()}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Right Pane: Active Step Notes + Safety Alerts + Deviations Log */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '20px'
      }}>
        {/* Step Note Pad */}
        <div className="glass-panel" style={{ padding: '20px' }}>
          <h4 style={{
            fontFamily: 'var(--font-title)',
            fontSize: '14px',
            fontWeight: 600,
            marginBottom: '12px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}>
            <ClipboardList size={16} />
            Step Notes
          </h4>
          <textarea
            className="glass-input"
            rows={4}
            placeholder="Type observations, measurements, or reagent batch details here..."
            value={stepNote}
            onChange={(e) => setStepNote(e.target.value)}
            disabled={updating}
            style={{ width: '100%', resize: 'none', fontSize: '13px', lineHeight: 1.4 }}
          />
          <button
            className="btn-secondary"
            onClick={handleSaveNote}
            disabled={updating}
            style={{ width: '100%', marginTop: '10px', padding: '8px' }}
          >
            {updating ? <span className="spinner" /> : 'Save Observation'}
          </button>
        </div>

        {/* Safety Requirements panel */}
        {currentActiveStepObj && (
          <div className="glass-panel" style={{ padding: '20px' }}>
            <h4 style={{
              fontFamily: 'var(--font-title)',
              fontSize: '14px',
              fontWeight: 600,
              marginBottom: '12px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              color: currentActiveStepObj.safety_level !== 'low' ? 'var(--warning)' : 'var(--text-primary)'
            }}>
              <Shield size={16} />
              Reagent Safety & Equipment
            </h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div>
                <p style={{ color: 'var(--text-secondary)', fontSize: '11px', fontWeight: 500 }}>REAGENTS IN USE</p>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: '4px' }}>
                  {currentActiveStepObj.reagents && currentActiveStepObj.reagents.length > 0 ? (
                    currentActiveStepObj.reagents.map((r, i) => (
                      <span key={i} style={{ background: 'rgba(255,255,255,0.04)', padding: '2px 6px', borderRadius: '4px', fontSize: '11px' }}>
                        {r}
                      </span>
                    ))
                  ) : (
                    <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>None specified</span>
                  )}
                </div>
              </div>

              <div style={{ marginTop: '4px' }}>
                <p style={{ color: 'var(--text-secondary)', fontSize: '11px', fontWeight: 500 }}>EQUIPMENT</p>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: '4px' }}>
                  {currentActiveStepObj.equipment && currentActiveStepObj.equipment.length > 0 ? (
                    currentActiveStepObj.equipment.map((e, i) => (
                      <span key={i} style={{ background: 'rgba(255,255,255,0.04)', padding: '2px 6px', borderRadius: '4px', fontSize: '11px' }}>
                        {e}
                      </span>
                    ))
                  ) : (
                    <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>None specified</span>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Deviations Log */}
        <div className="glass-panel" style={{
          padding: '20px',
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          maxHeight: '300px'
        }}>
          <h4 style={{
            fontFamily: 'var(--font-title)',
            fontSize: '14px',
            fontWeight: 600,
            marginBottom: '12px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}>
            <AlertOctagon size={16} />
            Deviation & Alerts Log
          </h4>
          <div style={{
            flex: 1,
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: '10px'
          }}>
            {(!experiment.deviations || experiment.deviations.length === 0) ? (
              <div style={{
                color: 'var(--text-muted)',
                fontSize: '12px',
                textAlign: 'center',
                marginTop: '30px'
              }}>
                No deviations detected. Good laboratory compliance!
              </div>
            ) : (
              experiment.deviations.map((dev, i) => (
                <div key={i} style={{
                  background: 'rgba(239, 68, 68, 0.03)',
                  border: '1px solid rgba(239, 68, 68, 0.15)',
                  borderRadius: 'var(--radius-md)',
                  padding: '8px 12px',
                  fontSize: '12px',
                  lineHeight: 1.4
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--danger)', fontWeight: 500, marginBottom: '2px' }}>
                    <span>Step {dev.step_number} Deviation</span>
                    <span style={{ fontSize: '10px', textTransform: 'uppercase' }}>{dev.severity}</span>
                  </div>
                  <p style={{ color: 'var(--text-secondary)' }}>{dev.description}</p>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
