import React, { useState } from 'react';
import { Upload, FileText, X } from 'lucide-react';
import { api } from '../services/api';

interface ProtocolUploaderProps {
  onUploadSuccess: (protocol: any) => void;
  onClose: () => void;
}

export const ProtocolUploader: React.FC<ProtocolUploaderProps> = ({ onUploadSuccess, onClose }) => {
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState('');
  const [experimentType, setExperimentType] = useState('PCR');
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type === 'application/pdf' || droppedFile.name.endsWith('.pdf')) {
        setFile(droppedFile);
        if (!title) {
          // Pre-populate title with file name minus extension
          setTitle(droppedFile.name.replace(/\.[^/.]+$/, ""));
        }
      } else {
        setError('Only PDF protocol files are supported.');
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      if (!title) {
        setTitle(selectedFile.name.replace(/\.[^/.]+$/, ""));
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a PDF file to upload.');
      return;
    }
    if (!title.trim()) {
      setError('Please specify a protocol title.');
      return;
    }

    setError(null);
    setUploading(true);

    try {
      const response = await api.uploadProtocol(file, title, experimentType);
      onUploadSuccess(response.protocol || response);
      onClose();
    } catch (err: any) {
      setError(err.message || 'Failed to upload and ingest the protocol.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.75)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 100,
      padding: '20px'
    }}>
      <div className="glass-panel animate-slide" style={{
        width: '100%',
        maxWidth: '500px',
        padding: '30px',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        position: 'relative'
      }}>
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '20px',
            right: '20px',
            background: 'none',
            border: 'none',
            color: 'var(--text-secondary)',
            cursor: 'pointer'
          }}
        >
          <X size={20} />
        </button>

        <h3 style={{
          fontFamily: 'var(--font-title)',
          fontSize: '18px',
          fontWeight: 600,
          marginBottom: '20px'
        }}>
          Upload Research Protocol
        </h3>

        {error && (
          <div className="animate-fade" style={{
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.2)',
            borderRadius: 'var(--radius-md)',
            color: 'var(--danger)',
            padding: '12px 16px',
            fontSize: '13px',
            marginBottom: '20px',
            lineHeight: 1.4
          }}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            style={{
              border: `2px dashed ${dragOver ? 'var(--primary)' : 'rgba(255, 255, 255, 0.15)'}`,
              borderRadius: 'var(--radius-lg)',
              padding: '40px 20px',
              textAlign: 'center',
              backgroundColor: dragOver ? 'rgba(99, 102, 241, 0.05)' : 'rgba(255, 255, 255, 0.01)',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '12px'
            }}
            onClick={() => document.getElementById('fileInput')?.click()}
          >
            <input
              id="fileInput"
              type="file"
              accept=".pdf,application/pdf"
              onChange={handleFileChange}
              style={{ display: 'none' }}
              disabled={uploading}
            />
            {file ? (
              <>
                <FileText size={40} style={{ color: 'var(--primary)' }} />
                <div>
                  <p style={{ fontWeight: 500, fontSize: '14px', wordBreak: 'break-all' }}>{file.name}</p>
                  <p style={{ color: 'var(--text-muted)', fontSize: '12px', marginTop: '4px' }}>
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              </>
            ) : (
              <>
                <Upload size={40} style={{ color: 'var(--text-secondary)' }} />
                <div>
                  <p style={{ fontWeight: 500, fontSize: '14px' }}>Drag & drop your protocol PDF here</p>
                  <p style={{ color: 'var(--text-muted)', fontSize: '12px', marginTop: '4px' }}>
                    or click to browse from files
                  </p>
                </div>
              </>
            )}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={{ color: 'var(--text-secondary)', fontSize: '12px', fontWeight: 500 }}>Protocol Title</label>
            <input
              className="glass-input"
              type="text"
              placeholder="e.g. Plasmid DNA Isolation"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              disabled={uploading}
              required
            />
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={{ color: 'var(--text-secondary)', fontSize: '12px', fontWeight: 500 }}>Experiment Type</label>
            <select
              className="glass-input"
              value={experimentType}
              onChange={(e) => setExperimentType(e.target.value)}
              disabled={uploading}
              style={{ background: 'var(--bg-surface)' }}
            >
              <option value="PCR">PCR (Polymerase Chain Reaction)</option>
              <option value="Gel Electrophoresis">Gel Electrophoresis</option>
              <option value="Transformation">Bacterial Transformation</option>
              <option value="DNA Extraction">DNA/RNA Extraction</option>
              <option value="Cloning">Molecular Cloning</option>
              <option value="Cell Culture">Cell Culture / Transfection</option>
              <option value="General">General / Other Wet Lab</option>
            </select>
          </div>

          <div style={{ display: 'flex', gap: '12px', marginTop: '10px' }}>
            <button
              className="btn-secondary"
              type="button"
              onClick={onClose}
              disabled={uploading}
              style={{ flex: 1 }}
            >
              Cancel
            </button>
            <button
              className="btn-primary"
              type="submit"
              disabled={uploading || !file}
              style={{ flex: 2 }}
            >
              {uploading ? <span className="spinner" /> : 'Ingest Protocol'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
