const envBaseUrl = import.meta.env.VITE_API_BASE_URL;
const rawBaseUrl = envBaseUrl ? envBaseUrl : 'http://localhost:8000/api/v1';
export const API_BASE_URL = rawBaseUrl.endsWith('/api/v1') ? rawBaseUrl : `${rawBaseUrl}/api/v1`;

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
  
  let responseText = '';
  try {
    responseText = await response.text();
  } catch (err) {
    // Ignore read errors
  }
  
  if (!response.ok) {
    let errorMessage = response.statusText || 'An error occurred';
    if (responseText) {
      try {
        const errorData = JSON.parse(responseText);
        if (Array.isArray(errorData.detail)) {
          errorMessage = errorData.detail.map((d: any) => `${d.loc.join('.')}: ${d.msg}`).join(', ');
        } else {
          errorMessage = errorData.detail || errorData.error?.message || errorMessage;
        }
      } catch {
        // Response is not JSON (e.g. HTML gateway error)
        errorMessage = responseText.slice(0, 150) || errorMessage;
      }
    }

    if (response.status === 401) {
      localStorage.removeItem('token');
    }
    throw new Error(errorMessage);
  }
  
  if (response.status === 204 || !responseText.trim()) {
    return {} as T;
  }
  
  try {
    return JSON.parse(responseText) as T;
  } catch (err) {
    throw new Error('Failed to parse server response');
  }
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
