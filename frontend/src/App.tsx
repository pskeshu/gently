import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Home from './pages/Home';
import CalibrationWizard from './pages/CalibrationWizard';
import VolumeAcquisition from './pages/VolumeAcquisition';
import SessionHistory from './pages/SessionHistory';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-900 text-white">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/calibration/:sessionId" element={<CalibrationWizard />} />
          <Route path="/volume/:sessionId" element={<VolumeAcquisition />} />
          <Route path="/history" element={<SessionHistory />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
