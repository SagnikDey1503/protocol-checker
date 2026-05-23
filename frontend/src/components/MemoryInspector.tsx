import React, { useState, useEffect } from 'react';
import { Search, Brain, Award, AlertCircle, Calendar, Sparkles } from 'lucide-react';
import { api } from '../services/api';

interface UserPattern {
  id: string;
  pattern_type: string;
  description: string;
  frequency: number;
  last_seen: string;
}

interface EpisodicMemory {
  id: string;
  episode_type: string;
  content: string;
  importance_score: number;
  created_at: string;
}

export const MemoryInspector: React.FC = () => {
  const [patterns, setPatterns] = useState<UserPattern[]>([]);
  const [episodes, setEpisodes] = useState<EpisodicMemory[]>([]);
  const [loading, setLoading] = useState(true);
  
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [searching, setSearching] = useState(false);

  useEffect(() => {
    async function loadMemoryData() {
      try {
        const profile = await api.getMemoryProfile();
        // Assuming profile returns { patterns: [...], episodes: [...], ... }
        setPatterns(profile.patterns || []);
        setEpisodes(profile.episodes || []);
      } catch (err) {
        console.error('Failed to load memory profile', err);
      } finally {
        setLoading(false);
      }
    }
    loadMemoryData();
  }, []);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    setSearching(true);
    try {
      const results = await api.recallMemory(searchQuery);
      setSearchResults(results.memories || results || []);
    } catch (err) {
      console.error('Failed to recall memory', err);
    } finally {
      setSearching(false);
    }
  };

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        color: 'var(--text-secondary)'
      }}>
        <span className="spinner" style={{ marginRight: '8px' }} />
        Retrieving cognitive memory modules...
      </div>
    );
  }

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '350px 1fr',
      gap: '20px',
      height: '100%',
      maxHeight: 'calc(100vh - 120px)'
    }}>
      {/* Left Pane: Interactive Semantic Recall search */}
      <div className="glass-panel" style={{
        padding: '24px',
        display: 'flex',
        flexDirection: 'column',
        gap: '20px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            background: 'var(--primary-glow)',
            color: 'var(--primary)',
            padding: '8px',
            borderRadius: '8px'
          }}>
            <Brain size={20} />
          </div>
          <h3 style={{ fontFamily: 'var(--font-title)', fontSize: '16px', fontWeight: 600 }}>
            Semantic Memory Recall
          </h3>
        </div>

        <p style={{ color: 'var(--text-secondary)', fontSize: '12px', lineHeight: 1.4 }}>
          Search across your vectorized long-term memory to retrieve relevant protocols, procedural concepts, and past observations.
        </p>

        <form onSubmit={handleSearch} style={{ display: 'flex', gap: '8px' }}>
          <input
            className="glass-input"
            type="text"
            placeholder="Search memory, e.g. PCR issues..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{ flex: 1, padding: '8px 12px', fontSize: '13px' }}
          />
          <button
            className="btn-primary"
            type="submit"
            disabled={searching || !searchQuery.trim()}
            style={{ padding: '8px 12px' }}
          >
            <Search size={16} />
          </button>
        </form>

        {/* Search Results */}
        <div style={{
          flex: 1,
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: '12px'
        }}>
          {searching ? (
            <div style={{ display: 'flex', justifyContent: 'center', marginTop: '20px', color: 'var(--text-muted)' }}>
              <span className="spinner" style={{ marginRight: '8px', width: '14px', height: '14px' }} />
              Recalling...
            </div>
          ) : searchResults.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', fontSize: '11px', textAlign: 'center', marginTop: '40px' }}>
              {searchQuery ? 'No matching memories found.' : 'Enter a query above to query semantic vector recall.'}
            </div>
          ) : (
            searchResults.map((res, i) => (
              <div key={i} className="animate-fade" style={{
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid var(--border-color)',
                borderRadius: 'var(--radius-md)',
                padding: '10px 12px',
                fontSize: '12px',
                lineHeight: 1.4
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--primary)', fontWeight: 500, fontSize: '11px', marginBottom: '4px' }}>
                  <span style={{ textTransform: 'capitalize' }}>{res.memory_type || 'observation'}</span>
                  <span>Score: {res.similarity_score ? (res.similarity_score * 100).toFixed(0) : '85'}%</span>
                </div>
                <p style={{ color: 'var(--text-secondary)' }}>{res.content || res.text || res}</p>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Right Pane: Behavioral Patterns & Episodic Log */}
      <div style={{
        display: 'grid',
        gridTemplateRows: '220px 1fr',
        gap: '20px',
        height: '100%',
        overflow: 'hidden'
      }}>
        {/* Behavioral Patterns */}
        <div className="glass-panel" style={{
          padding: '24px',
          display: 'flex',
          flexDirection: 'column'
        }}>
          <h3 style={{
            fontFamily: 'var(--font-title)',
            fontSize: '15px',
            fontWeight: 600,
            marginBottom: '16px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <Sparkles size={16} style={{ color: 'var(--warning)' }} />
            Extracted Behavioral Patterns & Habits
          </h3>

          <div style={{
            flex: 1,
            overflowY: 'auto',
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
            gap: '12px'
          }}>
            {patterns.length === 0 ? (
              <div style={{
                gridColumn: '1 / -1',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--text-muted)',
                fontSize: '13px'
              }}>
                No behavioral patterns established yet. Keep interacting to let the agent study preferences.
              </div>
            ) : (
              patterns.map((pat) => (
                <div key={pat.id} className="animate-fade" style={{
                  background: pat.pattern_type === 'common_mistake' ? 'rgba(239, 68, 68, 0.02)' : 'rgba(16, 185, 129, 0.02)',
                  border: `1px solid ${pat.pattern_type === 'common_mistake' ? 'rgba(239, 68, 68, 0.12)' : 'rgba(16, 185, 129, 0.12)'}`,
                  borderRadius: 'var(--radius-md)',
                  padding: '12px 14px',
                  display: 'flex',
                  gap: '10px'
                }}>
                  {pat.pattern_type === 'common_mistake' ? (
                    <AlertCircle size={18} style={{ color: 'var(--danger)', flexShrink: 0 }} />
                  ) : (
                    <Award size={18} style={{ color: 'var(--accent)', flexShrink: 0 }} />
                  )}
                  <div>
                    <h5 style={{ fontWeight: 600, fontSize: '13px', textTransform: 'capitalize', marginBottom: '2px' }}>
                      {pat.pattern_type.replace('_', ' ')}
                    </h5>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '12px', lineHeight: 1.4 }}>
                      {pat.description}
                    </p>
                    <span style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block', marginTop: '6px' }}>
                      Recorded frequency: {pat.frequency}x
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Episodic Memory Logs */}
        <div className="glass-panel" style={{
          padding: '24px',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden'
        }}>
          <h3 style={{
            fontFamily: 'var(--font-title)',
            fontSize: '15px',
            fontWeight: 600,
            marginBottom: '16px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <Calendar size={16} style={{ color: 'var(--primary)' }} />
            Key Episodic Logs & Learnings
          </h3>

          <div style={{
            flex: 1,
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: '12px'
          }}>
            {episodes.length === 0 ? (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                color: 'var(--text-muted)',
                fontSize: '13px'
              }}>
                No major episodic experiences saved yet. Mark experiments complete to log summaries.
              </div>
            ) : (
              episodes.map((ep) => (
                <div key={ep.id} className="animate-fade" style={{
                  background: 'rgba(255, 255, 255, 0.01)',
                  border: '1px solid var(--border-color)',
                  borderRadius: 'var(--radius-md)',
                  padding: '14px 18px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'flex-start',
                  gap: '16px'
                }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                      <span style={{
                        fontSize: '10px',
                        background: ep.episode_type === 'mistake' ? 'var(--danger-glow)' : ep.episode_type === 'success' ? 'var(--accent-glow)' : 'var(--primary-glow)',
                        color: ep.episode_type === 'mistake' ? 'var(--danger)' : ep.episode_type === 'success' ? 'var(--accent)' : 'var(--primary)',
                        padding: '1px 6px',
                        borderRadius: '4px',
                        fontWeight: 600,
                        textTransform: 'uppercase'
                      }}>
                        {ep.episode_type}
                      </span>
                      <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                        {new Date(ep.created_at).toLocaleDateString()}
                      </span>
                    </div>
                    <p style={{ color: 'var(--text-primary)', fontSize: '13px', lineHeight: 1.4 }}>
                      {ep.content}
                    </p>
                  </div>

                  <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'flex-end',
                    flexShrink: 0
                  }}>
                    <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Importance</span>
                    <span style={{ fontSize: '14px', fontWeight: 600, color: ep.importance_score >= 0.7 ? 'var(--danger)' : ep.importance_score >= 0.4 ? 'var(--warning)' : 'var(--text-secondary)' }}>
                      {(ep.importance_score * 10).toFixed(0)}/10
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
