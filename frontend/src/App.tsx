import React from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import { AuthScreen } from './components/AuthScreen';
import { Dashboard } from './components/Dashboard';

const AppContent: React.FC = () => {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        backgroundColor: 'var(--bg-base)',
        color: 'var(--text-secondary)'
      }}>
        <span className="spinner" style={{ width: '24px', height: '24px', borderWidth: '3px', marginBottom: '12px' }} />
        <span style={{ fontSize: '13px', fontWeight: 500 }}>Initializing Research Bench...</span>
      </div>
    );
  }

  return isAuthenticated ? <Dashboard /> : <AuthScreen />;
};

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App;
