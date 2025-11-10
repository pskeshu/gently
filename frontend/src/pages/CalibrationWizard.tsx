/**
 * CalibrationWizard - Multi-step workflow for embryo calibration
 *
 * Steps:
 * 1. Capture initial image
 * 2. Mark all embryos
 * 3. Calibrate each embryo (loop)
 * 4. Summary
 */

import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { sessionApi, hardwareApi, embryoApi, imageApi } from '../api/client';
import { useWebSocket } from '../hooks/useWebSocket';
import EmbryoMarker from '../components/EmbryoMarker';
import type {
  Session, Embryo, EmbryoMarker as Marker,
  WSEmbryoProgress, WSCalibrationComplete, CaptureImageResponse
} from '../types';

type WizardStep = 'capture' | 'mark' | 'calibrate' | 'summary';

export default function CalibrationWizard() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();

  const [session, setSession] = useState<Session | null>(null);
  const [currentStep, setCurrentStep] = useState<WizardStep>('capture');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Image state
  const [initialImage, setInitialImage] = useState<CaptureImageResponse | null>(null);
  const [capturingImage, setCapturingImage] = useState(false);

  // Marking state
  const [markers, setMarkers] = useState<Marker[]>([]);
  const [savingMarkers, setSavingMarkers] = useState(false);

  // Calibration state
  const [embryos, setEmbryos] = useState<Embryo[]>([]);
  const [currentEmbryoIndex, setCurrentEmbryoIndex] = useState(0);
  const [calibrationStatus, setCalibrationStatus] = useState<string>('');
  const [isCalibrating, setIsCalibrating] = useState<boolean>(false);
  const [verificationImage, setVerificationImage] = useState<string | null>(null);

  // WebSocket for real-time updates
  const { lastMessage } = useWebSocket('/ws/calibration', {
    onMessage: (msg) => {
      if (msg.type === 'embryo_progress') {
        const progress = msg as WSEmbryoProgress;
        setCalibrationStatus(`${progress.stage}...`);
      } else if (msg.type === 'calibration_complete') {
        const complete = msg as WSCalibrationComplete;
        if (complete.success) {
          // Reload embryo data
          loadEmbryos();
        }
      }
    },
  });

  useEffect(() => {
    loadSession();
    loadEmbryos();
  }, [sessionId]);

  const loadSession = async () => {
    try {
      const data = await sessionApi.get(Number(sessionId));
      setSession(data);
    } catch (err) {
      setError(`Failed to load session: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  const loadEmbryos = async () => {
    try {
      const data = await embryoApi.list(Number(sessionId), true);
      setEmbryos(data);
    } catch (err) {
      console.error('Failed to load embryos:', err);
    }
  };

  // Step 1: Capture initial image
  const handleCaptureImage = async () => {
    setCapturingImage(true);
    setError(null);

    try {
      const result = await hardwareApi.captureImage();
      if (!result.success) {
        throw new Error(result.error || 'Image capture failed');
      }

      setInitialImage(result);
      setCurrentStep('mark');
    } catch (err: any) {
      setError(`Image capture failed: ${err.message}`);
    } finally {
      setCapturingImage(false);
    }
  };

  // Step 2: Mark embryos
  const handleAddMarker = (x: number, y: number) => {
    const embryoNumber = markers.length + 1;
    setMarkers([...markers, { embryo_number: embryoNumber, x, y }]);
  };

  const handleRemoveMarker = (index: number) => {
    const updated = markers.filter((_, i) => i !== index);
    // Renumber embryos
    const renumbered = updated.map((m, i) => ({ ...m, embryo_number: i + 1 }));
    setMarkers(renumbered);
  };

  const handleDoneMarking = async () => {
    if (markers.length === 0) {
      setError('Please mark at least one embryo');
      return;
    }

    setSavingMarkers(true);
    setError(null);

    try {
      // Save all markers to database
      const savedEmbryos: Embryo[] = [];
      for (const marker of markers) {
        const embryo = await embryoApi.mark(
          Number(sessionId),
          marker.embryo_number,
          marker.x,
          marker.y,
          initialImage!.stage_position!.x,
          initialImage!.stage_position!.y
        );
        savedEmbryos.push(embryo);

        // Update marker with database ID
        marker.embryo_id = embryo.id;
      }

      // Store initial image for first embryo (or all?)
      if (savedEmbryos.length > 0 && initialImage?.image) {
        await imageApi.store(savedEmbryos[0].id, 'initial', initialImage.image);
      }

      setEmbryos(savedEmbryos);
      setCurrentStep('calibrate');
      setCurrentEmbryoIndex(0);
    } catch (err: any) {
      setError(`Failed to save markers: ${err.message}`);
    } finally {
      setSavingMarkers(false);
    }
  };

  // Step 3: Calibrate each embryo
  const handleCenterEmbryo = async () => {
    const embryo = embryos[currentEmbryoIndex];
    if (!embryo || !initialImage) return;

    setCalibrationStatus('Centering embryo...');
    setError(null);

    try {
      await embryoApi.center(
        embryo.id,
        initialImage.shape![0],
        initialImage.shape![1]
      );

      // Capture verification image
      setCalibrationStatus('Capturing verification image...');
      const verifyImg = await hardwareApi.captureImage();
      if (verifyImg.success && verifyImg.image) {
        setVerificationImage(verifyImg.image);
        await imageApi.store(embryo.id, 'centered', verifyImg.image);
      }

      setCalibrationStatus('');
    } catch (err: any) {
      setError(`Centering failed: ${err.message}`);
      setCalibrationStatus('');
    }
  };

  const handleRunCalibration = async () => {
    const embryo = embryos[currentEmbryoIndex];
    if (!embryo) return;

    setIsCalibrating(true);
    setCalibrationStatus('Running calibration...');
    setError(null);

    try {
      await embryoApi.calibrate(embryo.id);
      setCalibrationStatus('Calibration complete!');

      // Move to next embryo or summary
      if (currentEmbryoIndex < embryos.length - 1) {
        setTimeout(() => {
          setCurrentEmbryoIndex(currentEmbryoIndex + 1);
          setVerificationImage(null);
          setCalibrationStatus('');
          setIsCalibrating(false);
        }, 1500);
      } else {
        // All done!
        setTimeout(() => {
          setCurrentStep('summary');
          loadEmbryos(); // Reload with calibration data
          setIsCalibrating(false);
        }, 1500);
      }
    } catch (err: any) {
      setError(`Calibration failed: ${err.message}`);
      setIsCalibrating(false);
      setCalibrationStatus('');
    }
  };

  const handleSkipEmbryo = () => {
    if (currentEmbryoIndex < embryos.length - 1) {
      setCurrentEmbryoIndex(currentEmbryoIndex + 1);
      setVerificationImage(null);
      setCalibrationStatus('');
      setError(null);
    } else {
      setCurrentStep('summary');
      loadEmbryos();
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="spinner"></div>
        <span className="ml-3">Loading session...</span>
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

  const currentEmbryo = embryos[currentEmbryoIndex];
  const progressPercent = currentStep === 'capture' ? 0 :
                         currentStep === 'mark' ? 25 :
                         currentStep === 'calibrate' ? 50 + (currentEmbryoIndex / embryos.length) * 40 :
                         100;

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold">{session.name}</h1>
              <p className="text-gray-400 mt-1">Embryo Calibration Wizard</p>
            </div>
            <button
              onClick={() => navigate('/')}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
            >
              ← Back to Sessions
            </button>
          </div>

          {/* Progress bar */}
          <div className="mb-2">
            <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 progress-bar"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
            <p className="text-sm text-gray-400 mt-1">{progressPercent.toFixed(0)}% complete</p>
          </div>
        </div>

        {/* Error message */}
        {error && (
          <div className="mb-6 p-4 bg-red-900/30 border border-red-500 rounded-lg text-red-200">
            {error}
          </div>
        )}

        {/* Step content */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          {/* STEP 1: Capture initial image */}
          {currentStep === 'capture' && (
            <div className="text-center">
              <h2 className="text-2xl font-semibold mb-4">Step 1: Capture Initial Image</h2>
              <p className="text-gray-300 mb-6">
                Position the sample and capture an image from the bottom camera.
              </p>

              <button
                onClick={handleCaptureImage}
                disabled={capturingImage}
                className="px-8 py-4 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded-lg font-semibold text-lg"
              >
                {capturingImage ? (
                  <span className="flex items-center gap-2">
                    <div className="spinner"></div>
                    Capturing...
                  </span>
                ) : (
                  '📷 Capture Image'
                )}
              </button>
            </div>
          )}

          {/* STEP 2: Mark embryos */}
          {currentStep === 'mark' && initialImage && (
            <div>
              <h2 className="text-2xl font-semibold mb-4">Step 2: Mark All Embryos</h2>
              <p className="text-gray-300 mb-6">
                Click on each embryo's center to mark it. You can mark multiple embryos.
              </p>

              <div className="mb-6">
                <EmbryoMarker
                  imageData={initialImage.image!}
                  markers={markers}
                  onAddMarker={handleAddMarker}
                  onRemoveMarker={handleRemoveMarker}
                  disabled={savingMarkers}
                />
              </div>

              <div className="flex justify-between items-center">
                <button
                  onClick={() => setCurrentStep('capture')}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
                  disabled={savingMarkers}
                >
                  ← Retake Image
                </button>

                <div className="flex gap-3">
                  {markers.length > 0 && (
                    <button
                      onClick={() => handleRemoveMarker(markers.length - 1)}
                      className="px-4 py-2 bg-orange-600 hover:bg-orange-700 rounded"
                      disabled={savingMarkers}
                    >
                      Undo Last
                    </button>
                  )}
                  <button
                    onClick={handleDoneMarking}
                    disabled={markers.length === 0 || savingMarkers}
                    className="px-6 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded font-semibold"
                  >
                    {savingMarkers ? 'Saving...' : `Done (${markers.length} embryo${markers.length !== 1 ? 's' : ''})`}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* STEP 3: Calibrate each embryo */}
          {currentStep === 'calibrate' && currentEmbryo && (
            <div>
              <h2 className="text-2xl font-semibold mb-4">
                Step 3: Calibrating Embryo #{currentEmbryo.embryo_number} of {embryos.length}
              </h2>

              {/* Progress within calibration step */}
              <div className="mb-6 p-4 bg-gray-750 rounded">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm">Embryo {currentEmbryoIndex + 1} / {embryos.length}</span>
                  <span className="text-sm">{((currentEmbryoIndex / embryos.length) * 100).toFixed(0)}%</span>
                </div>
                <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-green-500 progress-bar"
                    style={{ width: `${(currentEmbryoIndex / embryos.length) * 100}%` }}
                  />
                </div>
              </div>

              {/* Verification image */}
              {verificationImage && (
                <div className="mb-6">
                  <h3 className="text-lg font-semibold mb-3">Centered Embryo:</h3>
                  <div className="flex justify-center">
                    <img
                      src={verificationImage}
                      alt="Verification"
                      className="max-w-2xl border-2 border-gray-700 rounded"
                    />
                  </div>
                </div>
              )}

              {/* Status */}
              {calibrationStatus && (
                <div className="mb-6 p-4 bg-blue-900/30 border border-blue-500 rounded">
                  <p className="text-blue-200">{calibrationStatus}</p>
                </div>
              )}

              {/* Calibration result */}
              {currentEmbryo.calibration_status === 'completed' && currentEmbryo.calibration && (
                <div className="mb-6 p-4 bg-green-900/30 border border-green-500 rounded">
                  <h3 className="text-lg font-semibold text-green-300 mb-3">✓ Calibration Complete</h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-400">Slope:</span>
                      <span className="ml-2 font-mono">{currentEmbryo.calibration.slope_um_per_deg.toFixed(3)} µm/°</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Offset:</span>
                      <span className="ml-2 font-mono">{currentEmbryo.calibration.offset_um.toFixed(2)} µm</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Galvo Top:</span>
                      <span className="ml-2 font-mono">{currentEmbryo.calibration.galvo_top_deg.toFixed(3)}°</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Galvo Bottom:</span>
                      <span className="ml-2 font-mono">{currentEmbryo.calibration.galvo_bottom_deg.toFixed(3)}°</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Actions */}
              <div className="flex justify-between">
                <button
                  onClick={handleSkipEmbryo}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
                >
                  Skip This Embryo
                </button>

                <div className="flex gap-3">
                  {!verificationImage && (
                    <button
                      onClick={handleCenterEmbryo}
                      className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded font-semibold"
                      disabled={isCalibrating}
                    >
                      Center Embryo
                    </button>
                  )}

                  {verificationImage && currentEmbryo.calibration_status !== 'completed' && (
                    <button
                      onClick={handleRunCalibration}
                      className="px-6 py-2 bg-green-600 hover:bg-green-700 rounded font-semibold"
                      disabled={isCalibrating}
                    >
                      Run Calibration
                    </button>
                  )}

                  {currentEmbryo.calibration_status === 'completed' && (
                    <button
                      onClick={handleSkipEmbryo}
                      className="px-6 py-2 bg-green-600 hover:bg-green-700 rounded font-semibold"
                    >
                      {currentEmbryoIndex < embryos.length - 1 ? 'Next Embryo →' : 'View Summary →'}
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* STEP 4: Summary */}
          {currentStep === 'summary' && (
            <div>
              <h2 className="text-2xl font-semibold mb-4">✓ Calibration Complete!</h2>
              <p className="text-gray-300 mb-6">
                All embryos have been calibrated. You can now proceed to volume acquisition.
              </p>

              {/* Embryo table */}
              <div className="mb-6 overflow-x-auto">
                <table className="w-full text-left">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="pb-3 pr-4">Embryo</th>
                      <th className="pb-3 pr-4">Position (µm)</th>
                      <th className="pb-3 pr-4">Status</th>
                      <th className="pb-3">Calibration</th>
                    </tr>
                  </thead>
                  <tbody>
                    {embryos.map((embryo) => (
                      <tr key={embryo.id} className="border-b border-gray-800">
                        <td className="py-3 pr-4 font-semibold">#{embryo.embryo_number}</td>
                        <td className="py-3 pr-4 font-mono text-sm">
                          {embryo.stage_position_centered ? (
                            `(${embryo.stage_position_centered.x.toFixed(1)}, ${embryo.stage_position_centered.y.toFixed(1)})`
                          ) : (
                            <span className="text-gray-500">Not centered</span>
                          )}
                        </td>
                        <td className="py-3 pr-4">
                          <span className={`px-2 py-1 rounded text-xs font-semibold ${
                            embryo.calibration_status === 'completed' ? 'bg-green-900/30 text-green-300' :
                            embryo.calibration_status === 'failed' ? 'bg-red-900/30 text-red-300' :
                            'bg-gray-700 text-gray-300'
                          }`}>
                            {embryo.calibration_status}
                          </span>
                        </td>
                        <td className="py-3">
                          {embryo.calibration ? (
                            <span className="font-mono text-sm">
                              {embryo.calibration.slope_um_per_deg.toFixed(3)} µm/°
                            </span>
                          ) : (
                            <span className="text-gray-500">—</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="flex justify-between">
                <button
                  onClick={() => navigate('/')}
                  className="px-6 py-2 bg-gray-700 hover:bg-gray-600 rounded font-semibold"
                >
                  ← Back to Sessions
                </button>
                <button
                  onClick={() => navigate(`/volume/${sessionId}`)}
                  className="px-6 py-2 bg-green-600 hover:bg-green-700 rounded font-semibold"
                >
                  Proceed to Volume Acquisition →
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
