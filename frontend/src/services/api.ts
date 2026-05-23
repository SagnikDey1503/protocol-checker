export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api/v1';

function getHeaders(isMultipart = false): HeadersInit {
  const token = localStorage.getItem('token');
  const headers: HeadersInit = {};
  
  if (!isMultipart) {
    headers['Content-Type'] = 'application/json';
  }
  
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  
  return headers;
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${path}`;
  const response = await fetch(url, options);
  
  if (!response.ok) {
    let errorMessage = 'An error occurred';
    try {
      const errorData = await response.json();
      if (Array.isArray(errorData.detail)) {
        errorMessage = errorData.detail.map((d: any) => `${d.loc.join('.')}: ${d.msg}`).join(', ');
      } else {
        errorMessage = errorData.detail || errorData.error?.message || errorMessage;
      }
    } catch {
      errorMessage = response.statusText || errorMessage;
    }

    if (response.status === 401) {
      localStorage.removeItem('token');
    }
    throw new Error(errorMessage);
  }
  
  if (response.status === 204) {
    return {} as T;
  }
  
  return response.json() as Promise<T>;
}

export const api = {
  // Auth
  async register(email: string, password: string, fullName: string) {
    return request<any>('/auth/register', {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify({ email, password, full_name: fullName }),
    });
  },

  async login(email: string, password: string) {
    return request<any>('/auth/login', {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify({ email, password }),
    });
  },

  async getMe() {
    const headers = getHeaders();
    return request<any>(`/auth/me?t=${Date.now()}`, {
      method: 'GET',
      headers: {
        ...headers,
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'Expires': '0',
      },
    });
  },

  // Protocols
  async uploadProtocol(file: File, title: string, experimentType: string) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('title', title);
    formData.append('experiment_type', experimentType);

    return request<any>('/protocols/upload', {
      method: 'POST',
      headers: getHeaders(true),
      body: formData,
    });
  },

  async getProtocols() {
    return request<any[]>('/protocols', {
      method: 'GET',
      headers: getHeaders(),
    });
  },

  async getProtocol(id: string) {
    return request<any>(`/protocols/${id}`, {
      method: 'GET',
      headers: getHeaders(),
    });
  },

  async getProtocolSteps(id: string) {
    return request<any>(`/protocols/${id}/steps`, {
      method: 'GET',
      headers: getHeaders(),
    });
  },

  async deleteProtocol(id: string) {
    return request<any>(`/protocols/${id}`, {
      method: 'DELETE',
      headers: getHeaders(),
    });
  },

  // Experiments
  async startExperiment(protocolId: string, title: string) {
    return request<any>('/experiments/start', {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify({ protocol_id: protocolId, title }),
    });
  },

  async updateExperimentStep(experimentId: string, stepNumber: number, notes?: string, deviation?: boolean) {
    return request<any>(`/experiments/${experimentId}/step`, {
      method: 'PUT',
      headers: getHeaders(),
      body: JSON.stringify({ step_number: stepNumber, notes, deviation: !!deviation }),
    });
  },

  async completeExperiment(experimentId: string) {
    return request<any>(`/experiments/${experimentId}/complete`, {
      method: 'POST',
      headers: getHeaders(),
    });
  },

  async getExperimentTimeline(experimentId: string) {
    return request<any[]>(`/experiments/${experimentId}/timeline`, {
      method: 'GET',
      headers: getHeaders(),
    });
  },

  async getExperimentStatus(experimentId: string) {
    return request<any>(`/experiments/${experimentId}`, {
      method: 'GET',
      headers: getHeaders(),
    });
  },

  async getExperiments() {
    return request<any[]>('/experiments', {
      method: 'GET',
      headers: getHeaders(),
    });
  },

  // Memory
  async getMemoryProfile() {
    return request<any>('/memory/profile', {
      method: 'GET',
      headers: getHeaders(),
    });
  },

  async recallMemory(query: string) {
    return request<any>('/memory/recall', {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify({ query }),
    });
  },
};
