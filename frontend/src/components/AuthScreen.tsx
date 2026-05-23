import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { Beaker } from 'lucide-react';

export const AuthScreen: React.FC = () => {
  const { login, register, authError, clearAuthError } = useAuth();
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    clearAuthError();
    setSubmitting(true);

    try {
      if (isLogin) {
        await login(email, password);
      } else {
        if (!fullName.trim()) {
          throw new Error('Full name is required.');
        }
        await register(email, password, fullName);
      }
    } catch (err: any) {
      setError(err.message || 'Authentication failed. Please check your credentials.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: '100vh',
      padding: '20px',
      position: 'relative',
      zIndex: 1
    }}>
      <div className="glass-panel animate-slide" style={{
        width: '100%',
        maxWidth: '440px',
        padding: '40px 30px',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        boxShadow: '0 20px 40px rgba(0, 0, 0, 0.6)'
      }}>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          marginBottom: '32px'
        }}>
          <div style={{
            background: 'linear-gradient(135deg, var(--primary), #8b5cf6)',
            padding: '12px',
            borderRadius: '12px',
            color: '#fff',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginBottom: '16px',
            boxShadow: '0 8px 16px var(--primary-glow)'
          }}>
            <Beaker size={28} />
          </div>
          <h1 style={{
            fontFamily: 'var(--font-title)',
            fontWeight: 700,
            fontSize: '24px',
            letterSpacing: '-0.02em',
            marginBottom: '6px',
            background: 'linear-gradient(90deg, #fff, var(--text-secondary))',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent'
          }}>
            Protocol Assistant
          </h1>
          <p style={{
            color: 'var(--text-secondary)',
            fontSize: '13px',
            textAlign: 'center'
          }}>
            Synthetic Biology & Wet Lab Research Companion
          </p>
        </div>

        {(error || authError) && (
          <div className="animate-fade" style={{
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.2)',
            borderRadius: 'var(--radius-md)',
            color: 'var(--danger)',
            padding: '12px 16px',
            fontSize: '13px',
            marginBottom: '20px',
            textAlign: 'left',
            lineHeight: 1.4
          }}>
            {error || authError}
          </div>
        )}

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {!isLogin && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <label style={{ color: 'var(--text-secondary)', fontSize: '12px', fontWeight: 500 }}>Full Name</label>
              <input
                className="glass-input"
                type="text"
                placeholder="Dr. Rosalind Franklin"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                disabled={submitting}
                required
              />
            </div>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={{ color: 'var(--text-secondary)', fontSize: '12px', fontWeight: 500 }}>Email Address</label>
            <input
              className="glass-input"
              type="email"
              placeholder="researcher@lab.edu"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={submitting}
              required
            />
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={{ color: 'var(--text-secondary)', fontSize: '12px', fontWeight: 500 }}>Password</label>
            <input
              className="glass-input"
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={submitting}
              minLength={8}
              required
            />
          </div>

          <button
            className="btn-primary"
            type="submit"
            disabled={submitting}
            style={{ marginTop: '12px', padding: '12px' }}
          >
            {submitting ? (
              <span className="spinner" />
            ) : isLogin ? (
              'Sign In'
            ) : (
              'Create Account'
            )}
          </button>
        </form>

        <div style={{
          marginTop: '24px',
          textAlign: 'center',
          fontSize: '13px',
          color: 'var(--text-muted)'
        }}>
          {isLogin ? "New to the platform? " : "Already have an account? "}
          <button
            onClick={() => {
              setIsLogin(!isLogin);
              setError(null);
              clearAuthError();
            }}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--primary)',
              cursor: 'pointer',
              fontWeight: 500,
              fontSize: '13px',
              padding: '0 4px',
              textDecoration: 'underline'
            }}
            disabled={submitting}
          >
            {isLogin ? 'Create one here' : 'Sign in here'}
          </button>
        </div>
      </div>
    </div>
  );
};
