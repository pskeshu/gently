/**
 * Session History Page
 *
 * View and manage all sessions (active and archived).
 */

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { sessionApi } from '../api/client';
import type { Session } from '../types';

export default function SessionHistory() {
  const navigate = useNavigate();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'active' | 'archived'>('all');

  useEffect(() => {
    loadSessions();
  }, [filter]);

  const loadSessions = async () => {
    setLoading(true);
    setError(null);

    try {
      const filterParam = filter === 'all' ? undefined : filter;
      const data = await sessionApi.list(filterParam);
      setSessions(data);
    } catch (err: any) {
      setError(`Failed to load sessions: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleArchive = async (sessionId: number) => {
    try {
      await sessionApi.updateStatus(sessionId, 'archived');
      loadSessions();
    } catch (err: any) {
      setError(`Failed to archive session: ${err.message}`);
    }
  };

  const handleUnarchive = async (sessionId: number) => {
    try {
      await sessionApi.updateStatus(sessionId, 'active');
      loadSessions();
    } catch (err: any) {
      setError(`Failed to unarchive session: ${err.message}`);
    }
  };

  const handleDelete = async (sessionId: number, sessionName: string) => {
    if (!confirm(`Delete session "${sessionName}" and all associated data?\n\nThis action cannot be undone.`)) {
      return;
    }

    try {
      await sessionApi.delete(sessionId);
      loadSessions();
    } catch (err: any) {
      setError(`Failed to delete session: ${err.message}`);
    }
  };

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold">Session History</h1>
              <p className="text-gray-400 mt-1">View and manage all calibration sessions</p>
            </div>
            <button
              onClick={() => navigate('/')}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
            >
              ← Back to Home
            </button>
          </div>

          {/* Filter tabs */}
          <div className="flex gap-2">
            <button
              onClick={() => setFilter('all')}
              className={`px-4 py-2 rounded ${
                filter === 'all'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              All
            </button>
            <button
              onClick={() => setFilter('active')}
              className={`px-4 py-2 rounded ${
                filter === 'active'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Active
            </button>
            <button
              onClick={() => setFilter('archived')}
              className={`px-4 py-2 rounded ${
                filter === 'archived'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Archived
            </button>
          </div>
        </div>

        {/* Error message */}
        {error && (
          <div className="mb-6 p-4 bg-red-900/30 border border-red-500 rounded-lg text-red-200">
            {error}
          </div>
        )}

        {/* Sessions table */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <div className="spinner"></div>
              <span className="ml-3">Loading sessions...</span>
            </div>
          ) : sessions.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-gray-400 mb-4">
                {filter === 'all' ? 'No sessions found' : `No ${filter} sessions`}
              </p>
              <button
                onClick={() => navigate('/')}
                className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded font-semibold"
              >
                Create New Session
              </button>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-700 bg-gray-750">
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Session
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Created
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Embryos
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Volume Runs
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800">
                  {sessions.map((session) => (
                    <tr key={session.id} className="hover:bg-gray-750 transition-colors">
                      <td className="px-6 py-4">
                        <div>
                          <div className="font-semibold">{session.name}</div>
                          {session.description && (
                            <div className="text-sm text-gray-400 mt-1">{session.description}</div>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-300">
                        {new Date(session.created_at).toLocaleString()}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-300">
                        {session.num_embryos}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-300">
                        {session.num_volume_runs}
                      </td>
                      <td className="px-6 py-4">
                        <span
                          className={`inline-flex px-2 py-1 text-xs font-semibold rounded ${
                            session.status === 'active'
                              ? 'bg-green-900/30 text-green-300'
                              : 'bg-gray-700 text-gray-300'
                          }`}
                        >
                          {session.status}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <button
                            onClick={() => navigate(`/calibration/${session.id}`)}
                            className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm"
                            title="Calibrate"
                          >
                            Calibrate
                          </button>
                          <button
                            onClick={() => navigate(`/volume/${session.id}`)}
                            className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-sm"
                            title="Acquire volumes"
                            disabled={session.num_embryos === 0}
                          >
                            Acquire
                          </button>
                          {session.status === 'active' ? (
                            <button
                              onClick={() => handleArchive(session.id)}
                              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm"
                              title="Archive session"
                            >
                              Archive
                            </button>
                          ) : (
                            <button
                              onClick={() => handleUnarchive(session.id)}
                              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm"
                              title="Unarchive session"
                            >
                              Unarchive
                            </button>
                          )}
                          <button
                            onClick={() => handleDelete(session.id, session.name)}
                            className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm"
                            title="Delete session"
                          >
                            Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Statistics */}
        {!loading && sessions.length > 0 && (
          <div className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div className="text-2xl font-bold">{sessions.length}</div>
              <div className="text-sm text-gray-400">Total Sessions</div>
            </div>
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div className="text-2xl font-bold">
                {sessions.reduce((sum, s) => sum + s.num_embryos, 0)}
              </div>
              <div className="text-sm text-gray-400">Total Embryos</div>
            </div>
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div className="text-2xl font-bold">
                {sessions.reduce((sum, s) => sum + s.num_volume_runs, 0)}
              </div>
              <div className="text-sm text-gray-400">Total Volume Runs</div>
            </div>
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div className="text-2xl font-bold">
                {sessions.filter((s) => s.status === 'active').length}
              </div>
              <div className="text-sm text-gray-400">Active Sessions</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
