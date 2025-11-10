/**
 * EmbryoMarker - Interactive HTML5 canvas for marking embryo positions
 *
 * Displays microscope image and allows clicking to mark embryo centers.
 * Shows numbered markers, center crosshair, and pixel offsets.
 */

import { useEffect, useRef, useState } from 'react';
import type { EmbryoMarker as Marker } from '../types';

interface EmbryoMarkerProps {
  imageData: string;  // base64 encoded image
  markers: Marker[];
  onAddMarker: (x: number, y: number) => void;
  onRemoveMarker: (index: number) => void;
  disabled?: boolean;
}

export default function EmbryoMarker({
  imageData,
  markers,
  onAddMarker,
  onRemoveMarker,
  disabled = false,
}: EmbryoMarkerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });
  const [hoveredMarker, setHoveredMarker] = useState<number | null>(null);

  // Load image from base64
  useEffect(() => {
    if (!imageData) return;

    const img = new Image();
    img.onload = () => {
      setImage(img);
      // Set canvas to image dimensions (scaled to fit container)
      const maxWidth = 1200;
      const maxHeight = 900;
      const scale = Math.min(maxWidth / img.width, maxHeight / img.height, 1);
      setCanvasSize({
        width: img.width * scale,
        height: img.height * scale,
      });
    };
    img.src = imageData;
  }, [imageData]);

  // Draw canvas whenever image or markers change
  useEffect(() => {
    if (!image || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw image
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    // Calculate scale factor from image to canvas
    const scaleX = canvas.width / image.width;
    const scaleY = canvas.height / image.height;

    // Draw center crosshair
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    ctx.strokeStyle = 'rgba(255, 0, 0, 0.7)';
    ctx.lineWidth = 2;
    ctx.setLineDash([10, 5]);
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, canvas.height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(canvas.width, centerY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw grid guidelines
    ctx.strokeStyle = 'rgba(0, 255, 255, 0.2)';
    ctx.lineWidth = 1;
    const gridLines = [0.2, 0.4, 0.6, 0.8];
    gridLines.forEach((ratio) => {
      // Vertical
      ctx.beginPath();
      ctx.moveTo(canvas.width * ratio, 0);
      ctx.lineTo(canvas.width * ratio, canvas.height);
      ctx.stroke();
      // Horizontal
      ctx.beginPath();
      ctx.moveTo(0, canvas.height * ratio);
      ctx.lineTo(canvas.width, canvas.height * ratio);
      ctx.stroke();
    });

    // Draw markers
    markers.forEach((marker, index) => {
      const x = marker.x * scaleX;
      const y = marker.y * scaleY;
      const isHovered = hoveredMarker === index;

      // Outer circle
      ctx.beginPath();
      ctx.arc(x, y, isHovered ? 22 : 18, 0, 2 * Math.PI);
      ctx.fillStyle = isHovered ? 'rgba(0, 255, 0, 0.3)' : 'rgba(0, 255, 0, 0.2)';
      ctx.fill();
      ctx.strokeStyle = isHovered ? 'rgba(255, 255, 255, 1)' : 'rgba(255, 255, 255, 0.8)';
      ctx.lineWidth = isHovered ? 4 : 3;
      ctx.stroke();

      // Inner circle
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(50, 205, 50, 0.9)';
      ctx.fill();

      // Crosshair
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x - 12, y);
      ctx.lineTo(x + 12, y);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x, y - 12);
      ctx.lineTo(x, y + 12);
      ctx.stroke();

      // Embryo number
      ctx.font = 'bold 16px Arial';
      ctx.fillStyle = 'yellow';
      ctx.strokeStyle = 'black';
      ctx.lineWidth = 3;
      const label = `#${marker.embryo_number}`;
      const textWidth = ctx.measureText(label).width;
      ctx.strokeText(label, x - textWidth / 2, y - 25);
      ctx.fillText(label, x - textWidth / 2, y - 25);

      // Offset text
      const offsetX = marker.x - image.width / 2;
      const offsetY = marker.y - image.height / 2;
      const offsetText = `(${offsetX > 0 ? '+' : ''}${offsetX.toFixed(0)}, ${offsetY > 0 ? '+' : ''}${offsetY.toFixed(0)}) px`;
      ctx.font = '12px monospace';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.8)';
      ctx.lineWidth = 2;
      const offsetWidth = ctx.measureText(offsetText).width;
      ctx.strokeText(offsetText, x - offsetWidth / 2, y + 35);
      ctx.fillText(offsetText, x - offsetWidth / 2, y + 35);
    });
  }, [image, markers, canvasSize, hoveredMarker]);

  // Handle canvas click
  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (disabled || !image || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;

    // Scale click coordinates to image coordinates
    const scaleX = image.width / canvas.width;
    const scaleY = image.height / canvas.height;
    const imageX = clickX * scaleX;
    const imageY = clickY * scaleY;

    // Check if clicked on existing marker (for removal)
    const clickedMarkerIndex = markers.findIndex((marker) => {
      const dx = marker.x - imageX;
      const dy = marker.y - imageY;
      const distance = Math.sqrt(dx * dx + dy * dy);
      return distance < 20; // 20 pixel radius for click detection
    });

    if (clickedMarkerIndex >= 0) {
      onRemoveMarker(clickedMarkerIndex);
    } else {
      onAddMarker(imageX, imageY);
    }
  };

  // Handle mouse move (for hover effect)
  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!image || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    // Scale to image coordinates
    const scaleX = image.width / canvas.width;
    const scaleY = image.height / canvas.height;
    const imageX = mouseX * scaleX;
    const imageY = mouseY * scaleY;

    // Find hovered marker
    const hoveredIndex = markers.findIndex((marker) => {
      const dx = marker.x - imageX;
      const dy = marker.y - imageY;
      const distance = Math.sqrt(dx * dx + dy * dy);
      return distance < 20;
    });

    setHoveredMarker(hoveredIndex >= 0 ? hoveredIndex : null);
  };

  return (
    <div className="flex flex-col items-center">
      <div className="relative inline-block border-2 border-gray-700 rounded-lg overflow-hidden bg-black">
        {!image ? (
          <div className="flex items-center justify-center w-full h-64 bg-gray-800">
            <div className="text-center">
              <div className="spinner mx-auto mb-3"></div>
              <p className="text-gray-400">Loading image...</p>
            </div>
          </div>
        ) : (
          <canvas
            ref={canvasRef}
            width={canvasSize.width}
            height={canvasSize.height}
            onClick={handleCanvasClick}
            onMouseMove={handleMouseMove}
            className={`${disabled ? 'cursor-not-allowed' : 'cursor-crosshair'}`}
            style={{ display: 'block' }}
          />
        )}

        {/* Overlay instructions */}
        {!disabled && image && (
          <div className="absolute top-4 left-4 bg-black/80 rounded px-3 py-2 text-sm max-w-xs">
            <p className="text-white font-semibold mb-1">Instructions:</p>
            <ul className="text-gray-300 text-xs space-y-1">
              <li>• Click to mark embryo center</li>
              <li>• Click marker to remove</li>
              <li>• Red lines = image center</li>
            </ul>
          </div>
        )}

        {/* Marker count */}
        {markers.length > 0 && (
          <div className="absolute bottom-4 right-4 bg-black/80 rounded px-3 py-2">
            <p className="text-white font-semibold">
              {markers.length} embryo{markers.length !== 1 ? 's' : ''} marked
            </p>
          </div>
        )}
      </div>

      {/* Image dimensions info */}
      {image && (
        <p className="mt-2 text-sm text-gray-400">
          Image: {image.width} × {image.height} px | Canvas: {canvasSize.width.toFixed(0)} × {canvasSize.height.toFixed(0)} px
        </p>
      )}
    </div>
  );
}
