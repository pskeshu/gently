/**
 * Home Page - Session Management
 *
 * Create new sessions, view recent sessions, navigate to workflows.
 */

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { sessionApi, hardwareApi } from '../api/client';
import type { Session, HardwareStatus } from '../types';

export default function Home() {
  const navigate = useNavigate();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [hardwareStatus, setHardwareStatus] = useState<HardwareStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newSessionName, setNewSessionName] = useState('');
  const [newSessionDescription, setNewSessionDescription] = useState('');

  useEffect(() => {
    loadSessions();
    loadHardwareStatus();
  }, []);

  const loadSessions = async () => {
    try {
      const data = await sessionApi.list('active');
      setSessions(data);
    } catch (err) {
      setError(`Failed to load sessions: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  const loadHardwareStatus = async () => {
    try {
      const status = await hardwareApi.getStatus();
      setHardwareStatus(status);
    } catch (err) {
      console.error('Failed to load hardware status:', err);
    }
  };

  const handleCreateSession = async () => {
    if (!newSessionName.trim()) {
      setError('Session name is required');
      return;
    }

    setCreating(true);
    setError(null);

    try {
      const session = await sessionApi.create(newSessionName, newSessionDescription);
      setShowCreateModal(false);
      setNewSessionName('');
      setNewSessionDescription('');
      navigate(`/calibration/${session.id}`);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to create session');
    } finally {
      setCreating(false);
    }
  };

  const handleDeleteSession = async (sessionId: number) => {
    if (!confirm('Delete this session and all associated data?')) {
      return;
    }

    try {
      await sessionApi.delete(sessionId);
      loadSessions();
    } catch (err) {
      setError(`Failed to delete session: ${err}`);
    }
  };

  return (
    <div className="min-h-screen p-6">
      {/* Header */}
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Multi-Embryo Calibration</h1>
          <p className="text-gray-400">Manage calibration sessions and volume acquisitions</p>
        </div>

        {/* Hardware Status */}
        {hardwareStatus && (
          <div className="mb-6 p-4 bg-gray-800 rounded-lg border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-gray-400">Hardware Status</h3>
                <div className="mt-2 flex items-center gap-4">
                  <span className={`inline-flex items-center gap-2 ${hardwareStatus.connected ? 'text-green-400' : 'text-red-400'}`}>
                    <span className={`w-2 h-2 rounded-full ${hardwareStatus.connected ? 'bg-green-400' : 'bg-red-400'}`}></span>
                    {hardwareStatus.connected ? 'Connected' : 'Disconnected'}
                  </span>
                  {hardwareStatus.connected && (
                    <span className="text-gray-300">
                      Stage: ({hardwareStatus.stage_position.x.toFixed(1)}, {hardwareStatus.stage_position.y.toFixed(1)}) µm
                    </span>
                  )}
                </div>
              </div>
              <button
                onClick={loadHardwareStatus}
                className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm"
              >
                Refresh
              </button>
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="mb-8 flex gap-4">
          <button
            onClick={() => setShowCreateModal(true)}
            className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold flex items-center gap-2"
          >
            <span>+ New Session</span>
          </button>
          <button
            onClick={() => navigate('/history')}
            className="px-6 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg font-semibold"
          >
            View History
          </button>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-900/30 border border-red-500 rounded-lg text-red-200">
            {error}
          </div>
        )}

        {/* Sessions List */}
        <div>
          <h2 className="text-2xl font-semibold mb-4">Active Sessions</h2>

          {loading ? (
            <div className="flex items-center justify-center py-12">
              <div className="spinner"></div>
              <span className="ml-3">Loading sessions...</span>
            </div>
          ) : sessions.length === 0 ? (
            <div className="text-center py-12 bg-gray-800 rounded-lg border border-gray-700">
              <p className="text-gray-400 mb-4">No active sessions</p>
              <button
                onClick={() => setShowCreateModal(true)}
                className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded font-semibold"
              >
                Create Your First Session
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {sessions.map((session) => (
                <div
                  key={session.id}
                  className="p-5 bg-gray-800 rounded-lg border border-gray-700 hover:border-blue-500 transition-colors"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h3 className="text-lg font-semibold">{session.name}</h3>
                      <p className="text-sm text-gray-400 mt-1">
                        {new Date(session.created_at).toLocaleDateString()}
                      </p>
                    </div>
                    <button
                      onClick={() => handleDeleteSession(session.id)}
                      className="text-red-400 hover:text-red-300 text-sm"
                      title="Delete session"
                    >
                      ✕
                    </button>
                  </div>

                  {session.description && (
                    <p className="text-sm text-gray-400 mb-4">{session.description}</p>
                  )}

                  <div className="flex items-center justify-between text-sm mb-4">
                    <span className="text-gray-400">Embryos: {session.num_embryos}</span>
                    <span className="text-gray-400">Volumes: {session.num_volume_runs}</span>
                  </div>

                  <div className="flex gap-2">
                    <button
                      onClick={() => navigate(`/calibration/${session.id}`)}
                      className="flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm font-semibold"
                    >
                      Calibrate
                    </button>
                    <button
                      onClick={() => navigate(`/volume/${session.id}`)}
                      className="flex-1 px-3 py-2 bg-green-600 hover:bg-green-700 rounded text-sm font-semibold"
                      disabled={session.num_embryos === 0}
                    >
                      Acquire
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Create Session Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full">
            <h2 className="text-2xl font-bold mb-4">Create New Session</h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Session Name <span className="text-red-400">*</span>
                </label>
                <input
                  type="text"
                  value={newSessionName}
                  onChange={(e) => setNewSessionName(e.target.value)}
                  placeholder="e.g., Sample_2025-01-15"
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:border-blue-500 focus:outline-none"
                  autoFocus
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">
                  Description (optional)
                </label>
                <textarea
                  value={newSessionDescription}
                  onChange={(e) => setNewSessionDescription(e.target.value)}
                  placeholder="Describe this experimental session..."
                  rows={3}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:border-blue-500 focus:outline-none"
                />
              </div>
            </div>

            <div className="mt-6 flex gap-3">
              <button
                onClick={() => {
                  setShowCreateModal(false);
                  setNewSessionName('');
                  setNewSessionDescription('');
                  setError(null);
                }}
                className="flex-1 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded font-semibold"
                disabled={creating}
              >
                Cancel
              </button>
              <button
                onClick={handleCreateSession}
                disabled={creating || !newSessionName.trim()}
                className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded font-semibold"
              >
                {creating ? 'Creating...' : 'Create Session'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
