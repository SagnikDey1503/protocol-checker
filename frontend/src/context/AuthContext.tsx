import React, { createContext, useContext, useState, useEffect } from 'react';
import { api } from '../services/api';

interface User {
  id: string;
  email: string;
  full_name: string;
  is_active: boolean;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  loading: boolean;
  authError: string | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, fullName: string) => Promise<void>;
  logout: () => void;
  clearAuthError: () => void;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'));
  const [loading, setLoading] = useState(true);
  const [authError, setAuthError] = useState<string | null>(null);

  useEffect(() => {
    async function loadUser() {
      if (token) {
        try {
          const userData = await api.getMe();
          setUser(userData);
        } catch (err) {
          console.error('Failed to load user profile, clearing session.', err);
          const message = err instanceof Error ? err.message : 'Your session is invalid or expired. Please log in again.';
          setAuthError(message);
          logout();
        }
      }
      setLoading(false);
    }
    loadUser();
  }, [token]);

  const login = async (email: string, password: string) => {
    setLoading(true);
    setAuthError(null);
    try {
      const data = await api.login(email, password);
      localStorage.setItem('token', data.access_token);
      setToken(data.access_token);
      setUser(data.user);
      setLoading(false);
    } catch (err) {
      logout();
      const message = err instanceof Error ? err.message : 'Authentication failed. Please check your credentials.';
      setAuthError(message);
      throw err;
    }
  };

  const register = async (email: string, password: string, fullName: string) => {
    setLoading(true);
    setAuthError(null);
    try {
      await api.register(email, password, fullName);
      // Auto login after registration
      await login(email, password);
    } catch (err) {
      setLoading(false);
      const message = err instanceof Error ? err.message : 'Registration failed. Please try again.';
      setAuthError(message);
      throw err;
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    setToken(null);
    setUser(null);
    setLoading(false);
  };

  const clearAuthError = () => setAuthError(null);

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        loading,
        authError,
        login,
        register,
        logout,
        clearAuthError,
        isAuthenticated: !!user,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
