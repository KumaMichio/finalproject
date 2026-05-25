import React from 'react'
import CameraFeed from './CameraFeed'

/**
 * Grid nhiều camera. Tự động chọn layout dựa trên số camera.
 */
export default function CameraGrid({ cameras, incidents, onCameraClick }) {
  const count = cameras.length
  const cols = count <= 1 ? 1 : count <= 4 ? 2 : 3

  // Tìm incident gần nhất cho mỗi camera
  const latestPerCam = {}
  for (const inc of incidents) {
    if (!latestPerCam[inc.camera_id]) latestPerCam[inc.camera_id] = inc
  }

  return (
    <div
      className="grid gap-3 h-full"
      style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
    >
      {cameras.map(cam => (
        <CameraFeed
          key={cam.id}
          camera={cam}
          latestIncident={latestPerCam[cam.id] ?? null}
          onClick={onCameraClick}
        />
      ))}
    </div>
  )
}
