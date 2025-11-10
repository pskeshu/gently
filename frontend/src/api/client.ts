/**
 * API client for communicating with the backend.
 */

import axios from 'axios';
import type {
  Session, Embryo, CaptureImageResponse, HardwareStatus,
  VolumeRun, VolumeAcquisition
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ============================================================================
// Session API
// ============================================================================

export const sessionApi = {
  /**
   * Create a new session
   */
  async create(name: string, description?: string): Promise<Session> {
    const response = await apiClient.post<Session>('/sessions', { name, description });
    return response.data;
  },

  /**
   * List all sessions
   */
  async list(status?: 'active' | 'archived'): Promise<Session[]> {
    const response = await apiClient.get<Session[]>('/sessions', {
      params: { status },
    });
    return response.data;
  },

  /**
   * Get session details
   */
  async get(sessionId: number): Promise<Session> {
    const response = await apiClient.get<Session>(`/sessions/${sessionId}`);
    return response.data;
  },

  /**
   * Update session status
   */
  async updateStatus(sessionId: number, status: 'active' | 'archived'): Promise<{ success: boolean }> {
    const response = await apiClient.put(`/sessions/${sessionId}/status`, null, {
      params: { status },
    });
    return response.data;
  },

  /**
   * Delete session
   */
  async delete(sessionId: number): Promise<{ success: boolean }> {
    const response = await apiClient.delete(`/sessions/${sessionId}`);
    return response.data;
  },
};

// ============================================================================
// Hardware API
// ============================================================================

export const hardwareApi = {
  /**
   * Get hardware status
   */
  async getStatus(): Promise<HardwareStatus> {
    const response = await apiClient.get<HardwareStatus>('/hardware/status');
    return response.data;
  },

  /**
   * Capture image from bottom camera
   */
  async captureImage(): Promise<CaptureImageResponse> {
    const response = await apiClient.post<CaptureImageResponse>('/hardware/capture');
    return response.data;
  },
};

// ============================================================================
// Embryo API
// ============================================================================

export const embryoApi = {
  /**
   * Mark an embryo at a pixel position
   */
  async mark(
    sessionId: number,
    embryoNumber: number,
    pixelX: number,
    pixelY: number,
    stageXInitial: number,
    stageYInitial: number
  ): Promise<Embryo> {
    const response = await apiClient.post<Embryo>('/embryos/mark', {
      session_id: sessionId,
      embryo_number: embryoNumber,
      pixel_x: pixelX,
      pixel_y: pixelY,
      stage_x_initial: stageXInitial,
      stage_y_initial: stageYInitial,
    });
    return response.data;
  },

  /**
   * List embryos (optionally filtered by session)
   */
  async list(sessionId?: number, includeCalibration: boolean = true): Promise<Embryo[]> {
    const response = await apiClient.get<Embryo[]>('/embryos', {
      params: { session_id: sessionId, include_calibration: includeCalibration },
    });
    return response.data;
  },

  /**
   * Get embryo details
   */
  async get(embryoId: number): Promise<Embryo> {
    const response = await apiClient.get<Embryo>(`/embryos/${embryoId}`);
    return response.data;
  },

  /**
   * Center embryo (move stage)
   */
  async center(embryoId: number, imageHeight: number, imageWidth: number): Promise<{
    success: boolean;
    target_position: { x: number; y: number };
    actual_position: { x: number; y: number };
    displacement_um: { x: number; y: number };
  }> {
    const response = await apiClient.post(`/embryos/${embryoId}/center`, {
      image_height: imageHeight,
      image_width: imageWidth,
    });
    return response.data;
  },

  /**
   * Run calibration for an embryo
   */
  async calibrate(embryoId: number): Promise<{
    success: boolean;
    embryo_id: number;
    calibration: any;
  }> {
    const response = await apiClient.post(`/embryos/${embryoId}/calibrate`);
    return response.data;
  },
};

// ============================================================================
// Image API
// ============================================================================

export const imageApi = {
  /**
   * Store an image
   */
  async store(embryoId: number, imageType: string, imageData: string): Promise<{ success: boolean; image_id: number }> {
    const response = await apiClient.post('/images', {
      embryo_id: embryoId,
      image_type: imageType,
      image_data: imageData,
    });
    return response.data;
  },

  /**
   * Get image data
   */
  async get(imageId: number): Promise<{
    id: number;
    embryo_id: number;
    image_type: string;
    image_data: string;
    timestamp: string;
  }> {
    const response = await apiClient.get(`/images/${imageId}`);
    return response.data;
  },
};

// ============================================================================
// Volume API
// ============================================================================

export const volumeApi = {
  /**
   * Create a new volume run
   */
  async createRun(
    sessionId: number,
    embryoIds: number[],
    numSlices: number,
    numTimepoints: number,
    intervalMinutes: number,
    name?: string
  ): Promise<VolumeRun> {
    const response = await apiClient.post<VolumeRun>('/volumes/runs', {
      session_id: sessionId,
      embryo_ids: embryoIds,
      num_slices: numSlices,
      num_timepoints: numTimepoints,
      interval_minutes: intervalMinutes,
      name,
    });
    return response.data;
  },

  /**
   * List volume runs (optionally filtered by session)
   */
  async listRuns(sessionId?: number): Promise<VolumeRun[]> {
    const response = await apiClient.get<VolumeRun[]>('/volumes/runs', {
      params: { session_id: sessionId },
    });
    return response.data;
  },

  /**
   * Get volume run details
   */
  async getRun(runId: number): Promise<VolumeRun> {
    const response = await apiClient.get<VolumeRun>(`/volumes/runs/${runId}`);
    return response.data;
  },
};

export default {
  session: sessionApi,
  hardware: hardwareApi,
  embryo: embryoApi,
  image: imageApi,
  volume: volumeApi,
};
