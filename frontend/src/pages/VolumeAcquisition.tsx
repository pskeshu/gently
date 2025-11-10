/**
 * Volume Acquisition Page
 *
 * Interface for running multi-embryo volume acquisitions with timelapse support.
 */

import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { sessionApi, embryoApi, volumeApi } from '../api/client';
import type { Session, Embryo } from '../types';

export default function VolumeAcquisition() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();

  const [session, setSession] = useState<Session | null>(null);
  const [embryos, setEmbryos] = useState<Embryo[]>([]);
  const [selectedEmbryoIds, setSelectedEmbryoIds] = useState<Set<number>>(new Set());

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Acquisition parameters
  const [numSlices, setNumSlices] = useState(50);
  const [numTimepoints, setNumTimepoints] = useState(1);
  const [intervalMinutes, setIntervalMinutes] = useState(2.0);
  const [runName, setRunName] = useState('');

  // Run state
  const [isRunning, setIsRunning] = useState(false);
  const [currentStatus, setCurrentStatus] = useState('');

  useEffect(() => {
    loadData();
  }, [sessionId]);

  const loadData = async () => {
    try {
      const [sessionData, embryoData] = await Promise.all([
        sessionApi.get(Number(sessionId)),
        embryoApi.list(Number(sessionId), true),
      ]);

      setSession(sessionData);
      setEmbryos(embryoData);

      // Auto-select all calibrated embryos
      const calibratedIds = new Set(
        embryoData
          .filter((e) => e.calibration_status === 'completed')
          .map((e) => e.id)
      );
      setSelectedEmbryoIds(calibratedIds);
    } catch (err: any) {
      setError(`Failed to load data: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleEmbryo = (embryoId: number) => {
    const newSet = new Set(selectedEmbryoIds);
    if (newSet.has(embryoId)) {
      newSet.delete(embryoId);
    } else {
      newSet.add(embryoId);
    }
    setSelectedEmbryoIds(newSet);
  };

  const handleSelectAll = () => {
    const calibratedIds = new Set(
      embryos
        .filter((e) => e.calibration_status === 'completed')
        .map((e) => e.id)
    );
    setSelectedEmbryoIds(calibratedIds);
  };

  const handleDeselectAll = () => {
    setSelectedEmbryoIds(new Set());
  };

  const calculateEstimatedTime = () => {
    const sliceTime = 0.05; // ~50ms per slice
    const overheadTime = 5; // ~5 seconds overhead per embryo
    const timePerEmbryo = numSlices * sliceTime + overheadTime;
    const totalTime = selectedEmbryoIds.size * timePerEmbryo * numTimepoints;
    const waitTime = (numTimepoints - 1) * intervalMinutes * 60;
    return totalTime + waitTime;
  };

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) {
      return `~${hours}h ${minutes}m`;
    }
    return `~${minutes}m`;
  };

  const handleStartAcquisition = async () => {
    if (selectedEmbryoIds.size === 0) {
      setError('Please select at least one embryo');
      return;
    }

    setIsRunning(true);
    setError(null);
    setCurrentStatus('Starting acquisition...');

    try {
      const embryoIdArray = Array.from(selectedEmbryoIds);
      const volumeRun = await volumeApi.createRun(
        Number(sessionId),
        embryoIdArray,
        numSlices,
        numTimepoints,
        intervalMinutes,
        runName || undefined
      );

      setCurrentStatus(`Volume run #${volumeRun.id} started!`);

      // TODO: Start background acquisition task via API
      // For now, just show success message

      setTimeout(() => {
        alert(`Volume acquisition started!\nRun ID: ${volumeRun.id}\n\nThe acquisition is running in the background.`);
        navigate('/');
      }, 2000);
    } catch (err: any) {
      setError(`Failed to start acquisition: ${err.message}`);
      setIsRunning(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="spinner"></div>
        <span className="ml-3">Loading...</span>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <p className="text-red-400 mb-4">Session not found</p>
          <button
            onClick={() => navigate('/')}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded"
          >
            Go Home
          </button>
        </div>
      </div>
    );
  }

  const calibratedEmbryos = embryos.filter((e) => e.calibration_status === 'completed');
  const estimatedTime = calculateEstimatedTime();

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold">{session.name}</h1>
              <p className="text-gray-400 mt-1">Multi-Embryo Volume Acquisition</p>
            </div>
            <button
              onClick={() => navigate('/')}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
            >
              ← Back to Sessions
            </button>
          </div>
        </div>

        {/* Error message */}
        {error && (
          <div className="mb-6 p-4 bg-red-900/30 border border-red-500 rounded-lg text-red-200">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: Embryo selection */}
          <div className="lg:col-span-2">
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold">Select Embryos</h2>
                <div className="flex gap-2">
                  <button
                    onClick={handleSelectAll}
                    className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm"
                    disabled={isRunning}
                  >
                    Select All
                  </button>
                  <button
                    onClick={handleDeselectAll}
                    className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm"
                    disabled={isRunning}
                  >
                    Deselect All
                  </button>
                </div>
              </div>

              {calibratedEmbryos.length === 0 ? (
                <div className="text-center py-8 text-gray-400">
                  <p className="mb-4">No calibrated embryos found</p>
                  <button
                    onClick={() => navigate(`/calibration/${sessionId}`)}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded"
                  >
                    Calibrate Embryos
                  </button>
                </div>
              ) : (
                <div className="space-y-2">
                  {calibratedEmbryos.map((embryo) => (
                    <label
                      key={embryo.id}
                      className={`flex items-start p-4 border rounded cursor-pointer transition-colors ${
                        selectedEmbryoIds.has(embryo.id)
                          ? 'border-blue-500 bg-blue-900/20'
                          : 'border-gray-700 hover:border-gray-600'
                      } ${isRunning ? 'cursor-not-allowed opacity-50' : ''}`}
                    >
                      <input
                        type="checkbox"
                        checked={selectedEmbryoIds.has(embryo.id)}
                        onChange={() => handleToggleEmbryo(embryo.id)}
                        disabled={isRunning}
                        className="mt-1 mr-3"
                      />
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <span className="font-semibold">Embryo #{embryo.embryo_number}</span>
                          <span className="text-green-400 text-sm">✓ Calibrated</span>
                        </div>
                        <div className="mt-2 grid grid-cols-2 gap-2 text-sm text-gray-400">
                          <div>
                            <span>Position:</span>
                            <span className="ml-2 font-mono">
                              ({embryo.stage_position_centered?.x.toFixed(1)}, {embryo.stage_position_centered?.y.toFixed(1)}) µm
                            </span>
                          </div>
                          {embryo.calibration && (
                            <div>
                              <span>Slope:</span>
                              <span className="ml-2 font-mono">
                                {embryo.calibration.slope_um_per_deg.toFixed(3)} µm/°
                              </span>
                            </div>
                          )}
                        </div>
                      </div>
                    </label>
                  ))}
                </div>
              )}

              <div className="mt-4 p-3 bg-gray-750 rounded">
                <p className="text-sm text-gray-300">
                  <span className="font-semibold">{selectedEmbryoIds.size}</span> embryo{selectedEmbryoIds.size !== 1 ? 's' : ''} selected
                </p>
              </div>
            </div>
          </div>

          {/* Right: Parameters */}
          <div className="space-y-6">
            {/* Acquisition parameters */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h2 className="text-xl font-semibold mb-4">Parameters</h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Run Name (optional)</label>
                  <input
                    type="text"
                    value={runName}
                    onChange={(e) => setRunName(e.target.value)}
                    placeholder="e.g., Overnight timelapse"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:border-blue-500 focus:outline-none"
                    disabled={isRunning}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Number of Slices</label>
                  <input
                    type="number"
                    value={numSlices}
                    onChange={(e) => setNumSlices(Number(e.target.value))}
                    min="1"
                    max="500"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:border-blue-500 focus:outline-none"
                    disabled={isRunning}
                  />
                  <p className="text-xs text-gray-400 mt-1">Z-stack depth per volume</p>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Number of Timepoints</label>
                  <input
                    type="number"
                    value={numTimepoints}
                    onChange={(e) => setNumTimepoints(Number(e.target.value))}
                    min="1"
                    max="1000"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:border-blue-500 focus:outline-none"
                    disabled={isRunning}
                  />
                  <p className="text-xs text-gray-400 mt-1">1 = single acquisition, &gt;1 = timelapse</p>
                </div>

                {numTimepoints > 1 && (
                  <div>
                    <label className="block text-sm font-medium mb-2">Interval (minutes)</label>
                    <input
                      type="number"
                      value={intervalMinutes}
                      onChange={(e) => setIntervalMinutes(Number(e.target.value))}
                      min="0.5"
                      step="0.5"
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:border-blue-500 focus:outline-none"
                      disabled={isRunning}
                    />
                    <p className="text-xs text-gray-400 mt-1">Time between timepoints</p>
                  </div>
                )}
              </div>
            </div>

            {/* Summary */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h2 className="text-xl font-semibold mb-4">Summary</h2>

              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Embryos:</span>
                  <span className="font-semibold">{selectedEmbryoIds.size}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Slices per volume:</span>
                  <span className="font-semibold">{numSlices}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Timepoints:</span>
                  <span className="font-semibold">{numTimepoints}</span>
                </div>
                {numTimepoints > 1 && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Interval:</span>
                    <span className="font-semibold">{intervalMinutes} min</span>
                  </div>
                )}
                <div className="pt-3 border-t border-gray-700 flex justify-between">
                  <span className="text-gray-400">Total acquisitions:</span>
                  <span className="font-semibold">{selectedEmbryoIds.size * numTimepoints}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Estimated time:</span>
                  <span className="font-semibold">{formatTime(estimatedTime)}</span>
                </div>
              </div>
            </div>

            {/* Start button */}
            <button
              onClick={handleStartAcquisition}
              disabled={isRunning || selectedEmbryoIds.size === 0}
              className="w-full px-6 py-4 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded-lg font-semibold text-lg"
            >
              {isRunning ? (
                <span className="flex items-center justify-center gap-2">
                  <div className="spinner"></div>
                  {currentStatus}
                </span>
              ) : (
                `▶ Start Acquisition`
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
