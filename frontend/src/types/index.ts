/**
 * TypeScript type definitions for the Multi-Embryo Calibration frontend.
 */

export interface Session {
  id: number;
  name: string;
  description?: string;
  created_at: string;
  status: 'active' | 'archived';
  num_embryos: number;
  num_volume_runs: number;
}

export interface Embryo {
  id: number;
  session_id: number;
  embryo_number: number;
  embryo_id: string;
  pixel_position?: {
    x: number;
    y: number;
  };
  stage_position_initial?: {
    x: number;
    y: number;
  };
  stage_position_centered?: {
    x: number;
    y: number;
  };
  calibration_status: 'pending' | 'calibrating' | 'completed' | 'failed';
  created_at: string;
  num_images: number;
  calibration?: CalibrationData;
}

export interface CalibrationData {
  slope_um_per_deg: number;
  offset_um: number;
  galvo_top_deg: number;
  galvo_bottom_deg: number;
  piezo_top_um: number;
  piezo_bottom_um: number;
  [key: string]: any;
}

export interface CaptureImageResponse {
  success: boolean;
  image?: string;  // base64 encoded
  shape?: [number, number];
  stage_position?: {
    x: number;
    y: number;
  };
  timestamp?: number;
  error?: string;
}

export interface HardwareStatus {
  connected: boolean;
  stage_position: {
    x: number;
    y: number;
  };
  bottom_camera: string;
  spim_camera: string;
  timestamp: number;
  error?: string;
}

export interface VolumeRun {
  id: number;
  session_id: number;
  name?: string;
  num_slices: number;
  num_timepoints: number;
  interval_minutes: number;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  started_at?: string;
  completed_at?: string;
  output_dir?: string;
  num_acquisitions: number;
  successful_acquisitions: number;
}

export interface VolumeAcquisition {
  id: number;
  volume_run_id: number;
  embryo_id: number;
  embryo_number?: number;
  timepoint: number;
  success: boolean;
  filename?: string;
  shape?: [number, number, number];
  timestamp: string;
  error_message?: string;
}

// WebSocket message types
export interface WSMessage {
  type: string;
  timestamp: number;
}

export interface WSEmbryoProgress extends WSMessage {
  type: 'embryo_progress';
  embryo_id: number;
  embryo_number: number;
  total_embryos?: number;
  stage: 'marking' | 'centering' | 'calibrating' | 'moving' | 'configuring' | 'acquiring' | 'saving' | 'centered';
  stage_position?: {
    x: number;
    y: number;
  };
}

export interface WSSliceProgress extends WSMessage {
  type: 'slice_progress';
  current_slice: number;
  total_slices: number;
  percentage: number;
}

export interface WSTimepointComplete extends WSMessage {
  type: 'timepoint_complete';
  timepoint: number;
  total_timepoints: number;
  next_timepoint_at?: string;
}

export interface WSCalibrationComplete extends WSMessage {
  type: 'calibration_complete';
  embryo_id: number;
  embryo_number?: number;
  success: boolean;
  calibration_data?: CalibrationData;
  error?: string;
}

export interface WSVolumeRunStatus extends WSMessage {
  type: 'volume_run_status';
  volume_run_id: number;
  status: string;
  message: string;
}

export interface WSError extends WSMessage {
  type: 'error';
  error: string;
  details?: string;
}

// Marker for embryo marking on canvas
export interface EmbryoMarker {
  embryo_number: number;
  x: number;  // pixel coordinates
  y: number;
  embryo_id?: number;  // database ID (after saving)
}
