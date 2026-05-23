import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { api } from '../services/api';
import { ProtocolUploader } from './ProtocolUploader';
import { ChatPanel } from './ChatPanel';
import { ExperimentTimeline } from './ExperimentTimeline';
import { MemoryInspector } from './MemoryInspector';
import {
  FileText,
  Plus,
  Brain,
  LogOut,
  User,
  FlaskConical,
  Beaker,
  ChevronRight
} from 'lucide-react';

interface Protocol {
  id: string;
  title: string;
  experiment_type: string;
  original_filename: string;
  step_count: number;
  is_processed: boolean;
  reagents?: string[];
  equipment?: string[];
}

export const Dashboard: React.FC = () => {
  const { user, logout } = useAuth();
  const [activeView, setActiveView] = useState<'run' | 'memory' | 'welcome'>('welcome');
  
  const [protocols, setProtocols] = useState<Protocol[]>([]);
  const [loadingProtocols, setLoadingProtocols] = useState(true);
  const [showUploader, setShowUploader] = useState(false);
  
  const [selectedProtocol, setSelectedProtocol] = useState<Protocol | null>(null);
  const [activeExperiment, setActiveExperiment] = useState<any | null>(null);
  const [protocolSteps, setProtocolSteps] = useState<any[]>([]);

  // Load user's protocols on mount
  const fetchProtocols = async () => {
    try {
      const response = await api.getProtocols();
      // Handle response structure (response.protocols or response directly)
      setProtocols(Array.isArray(response) ? response : (response as any).protocols || []);
    } catch (err) {
      console.error('Failed to load protocols', err);
    } finally {
      setLoadingProtocols(false);
    }
  };

  useEffect(() => {
    fetchProtocols();
  }, []);

  const handleProtocolSelect = (protocol: Protocol) => {
    setSelectedProtocol(protocol);
    setActiveView('welcome');
    setActiveExperiment(null);
    setProtocolSteps([]);
  };

  const handleStartExperiment = async () => {
    if (!selectedProtocol) return;

    try {
      const newSession = await api.startExperiment(
        selectedProtocol.id,
        `${selectedProtocol.title} Run - ${new Date().toLocaleDateString()}`
      );
      
      // Fetch protocol steps
      const stepsResponse = await api.getProtocolSteps(selectedProtocol.id);
      setProtocolSteps(stepsResponse.steps || []);
      
      setActiveExperiment(newSession);
      setActiveView('run');
    } catch (err) {
      console.error('Failed to start experiment session', err);
      alert('Could not start experiment. Please ensure the protocol has finished background parsing.');
    }
  };

  const handleUploadSuccess = () => {
    fetchProtocols();
  };

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'var(--sidebar-width) 1fr',
      height: '100vh',
      width: '100vw',
      overflow: 'hidden'
    }}>
      {/* Sidebar Section */}
      <div style={{
        background: 'var(--bg-surface)',
        borderRight: '1px solid var(--border-color)',
        display: 'flex',
        flexDirection: 'column',
        height: '100%'
      }}>
        {/* Sidebar Header */}
        <div style={{
          padding: '24px 20px',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          alignItems: 'center',
          gap: '10px'
        }}>
          <div style={{
            background: 'var(--primary-glow)',
            color: 'var(--primary)',
            padding: '6px',
            borderRadius: '6px'
          }}>
            <Beaker size={20} />
          </div>
          <span style={{
            fontFamily: 'var(--font-title)',
            fontWeight: 700,
            fontSize: '16px',
            color: '#fff'
          }}>
            Bio Lab Bench
          </span>
        </div>

        {/* User profile details tag */}
        <div style={{
          padding: '16px 20px',
          background: 'rgba(255, 255, 255, 0.01)',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
            <div style={{
              width: '28px',
              height: '28px',
              borderRadius: '50%',
              backgroundColor: 'rgba(255, 255, 255, 0.05)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--text-secondary)'
            }}>
              <User size={14} />
            </div>
            <span style={{
              fontWeight: 500,
              fontSize: '13px',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              color: 'var(--text-primary)'
            }}>
              {user?.full_name || 'Researcher'}
            </span>
          </div>

          <button
            onClick={logout}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center'
            }}
          >
            <LogOut size={16} />
          </button>
        </div>

        {/* Action views router */}
        <div style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <button
            onClick={() => setActiveView('memory')}
            style={{
              width: '100%',
              background: activeView === 'memory' ? 'var(--primary-glow)' : 'transparent',
              border: 'none',
              borderRadius: 'var(--radius-md)',
              color: activeView === 'memory' ? 'var(--primary)' : 'var(--text-secondary)',
              padding: '10px 12px',
              cursor: 'pointer',
              fontWeight: 500,
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              fontSize: '13px',
              textAlign: 'left'
            }}
          >
            <Brain size={16} />
            Cognitive Memory Profile
          </button>
        </div>

        {/* Protocols list header */}
        <div style={{
          padding: '12px 20px 6px 20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600 }}>PROTOCOLS</span>
          <button
            onClick={() => setShowUploader(true)}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--primary)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center'
            }}
          >
            <Plus size={16} />
          </button>
        </div>

        {/* Protocols list */}
        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '4px 14px',
          display: 'flex',
          flexDirection: 'column',
          gap: '4px'
        }}>
          {loadingProtocols ? (
            <div style={{ display: 'flex', justifyContent: 'center', color: 'var(--text-muted)', marginTop: '20px' }}>
              <span className="spinner" style={{ width: '12px', height: '12px', marginRight: '6px' }} />
              Loading templates...
            </div>
          ) : protocols.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', fontSize: '11px', textAlign: 'center', marginTop: '20px' }}>
              No templates uploaded yet. Click '+' to upload.
            </div>
          ) : (
            protocols.map((protocol) => {
              const isSelected = selectedProtocol?.id === protocol.id;
              return (
                <button
                  key={protocol.id}
                  onClick={() => handleProtocolSelect(protocol)}
                  style={{
                    width: '100%',
                    background: isSelected ? 'rgba(255,255,255,0.03)' : 'transparent',
                    border: isSelected ? '1px solid rgba(255,255,255,0.06)' : '1px solid transparent',
                    borderRadius: 'var(--radius-md)',
                    color: isSelected ? '#fff' : 'var(--text-secondary)',
                    padding: '10px 12px',
                    cursor: 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '2px',
                    textAlign: 'left'
                  }}
                >
                  <span style={{ fontWeight: 500, fontSize: '13px', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap', width: '100%' }}>
                    {protocol.title}
                  </span>
                  <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', fontSize: '10px', color: 'var(--text-muted)' }}>
                    <span>{protocol.experiment_type}</span>
                    <span>{protocol.is_processed ? `${protocol.step_count} steps` : 'Processing...'}</span>
                  </div>
                </button>
              );
            })
          )}
        </div>
      </div>

      {/* Main Content Workspace Pane */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        padding: '24px',
        overflow: 'hidden'
      }}>
        {activeView === 'welcome' && selectedProtocol && (
          <div className="glass-panel animate-slide" style={{
            padding: '30px',
            maxWidth: '650px',
            margin: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: '20px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{ background: 'var(--primary-glow)', color: 'var(--primary)', padding: '10px', borderRadius: '10px' }}>
                <FileText size={24} />
              </div>
              <div>
                <h2 style={{ fontFamily: 'var(--font-title)', fontWeight: 600, fontSize: '20px' }}>
                  {selectedProtocol.title}
                </h2>
                <p style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>
                  Category: {selectedProtocol.experiment_type}
                </p>
              </div>
            </div>

            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '16px',
              borderTop: '1px solid var(--border-color)',
              borderBottom: '1px solid var(--border-color)',
              padding: '16px 0'
            }}>
              <div>
                <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
                  REAGENTS IDENTIFIED
                </span>
                <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                  {selectedProtocol.reagents && selectedProtocol.reagents.length > 0
                    ? selectedProtocol.reagents.join(', ')
                    : 'None specified'}
                </p>
              </div>
              <div>
                <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
                  EQUIPMENT DETECTED
                </span>
                <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                  {selectedProtocol.equipment && selectedProtocol.equipment.length > 0
                    ? selectedProtocol.equipment.join(', ')
                    : 'None specified'}
                </p>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '10px' }}>
              <div>
                <p style={{ fontWeight: 500, fontSize: '14px' }}>Structure verified</p>
                <p style={{ color: 'var(--text-muted)', fontSize: '12px' }}>Ready to launch interactive run.</p>
              </div>
              <button
                className="btn-primary"
                onClick={handleStartExperiment}
                disabled={!selectedProtocol.is_processed}
              >
                Start Experiment
                <ChevronRight size={16} />
              </button>
            </div>
          </div>
        )}

        {activeView === 'welcome' && !selectedProtocol && (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            color: 'var(--text-secondary)',
            textAlign: 'center'
          }}>
            <FlaskConical size={60} style={{ color: 'var(--border-color-hover)', marginBottom: '20px' }} />
            <h2 style={{ fontFamily: 'var(--font-title)', fontWeight: 600, fontSize: '20px', color: '#fff', marginBottom: '8px' }}>
              Welcome to the AI Lab Bench
            </h2>
            <p style={{ fontSize: '13px', maxWidth: '400px' }}>
              Select a research protocol template from the left sidebar or upload a new experimental procedure to start your assistant companion session.
            </p>
          </div>
        )}

        {activeView === 'run' && activeExperiment && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 400px',
            gap: '24px',
            height: '100%',
            overflow: 'hidden'
          }}>
            {/* Timeline progression track */}
            <div style={{ height: '100%', overflow: 'hidden' }}>
              <ExperimentTimeline
                experiment={activeExperiment}
                protocolSteps={protocolSteps}
                onStepChange={(updated) => setActiveExperiment(updated)}
              />
            </div>

            {/* Chat companion terminal */}
            <div style={{ height: '100%', overflow: 'hidden' }}>
              <ChatPanel
                sessionId={activeExperiment.id} // session_id maps directly to active experiment ID
                experimentId={activeExperiment.id}
              />
            </div>
          </div>
        )}

        {activeView === 'memory' && (
          <div style={{ height: '100%', overflow: 'hidden' }}>
            <MemoryInspector />
          </div>
        )}
      </div>

      {/* Uploader overlay modal */}
      {showUploader && (
        <ProtocolUploader
          onUploadSuccess={handleUploadSuccess}
          onClose={() => setShowUploader(false)}
        />
      )}
    </div>
  );
};
