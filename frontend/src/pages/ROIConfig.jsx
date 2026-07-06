import React, { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../services/api'

// UC4.2 — Nguoi van hanh chinh vung quan tam (ROI) truc tiep tren UI:
//  - kind='alert_zone': vung canh bao ra/vao (entry/exit/loiter/speed)
//  - kind='lane':       vung lan cho phat hien DI SAI LAN (allowed_classes,
//                       allowed_direction)
// Ve polygon bang cach click len anh snapshot cua camera. Toa do luu theo
// DO PHAN GIAI GOC cua camera (pipeline lam viec o resolution goc), nen SVG
// dung viewBox = kich thuoc goc de map click -> native chinh xac.

const VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck', 'person']
const DIRECTIONS = [
  { label: 'Không ràng buộc', value: null },
  { label: '→ Phải',   value: [1, 0] },
  { label: '← Trái',   value: [-1, 0] },
  { label: '↓ Xuống',  value: [0, 1] },
  { label: '↑ Lên',    value: [0, -1] },
  { label: '↘',        value: [1, 1] },
  { label: '↙',        value: [-1, 1] },
  { label: '↗',        value: [1, -1] },
  { label: '↖',        value: [-1, -1] },
]

const emptyDraft = () => ({
  name: '',
  kind: 'lane',
  points: [],                 // [[x,y], ...] toa do goc camera
  alertTypes: ['entry'],
  allowedClasses: ['car'],
  allowedDirIdx: 0,
})

export default function ROIConfig() {
  const navigate = useNavigate()
  const [cameras, setCameras] = useState([])
  const [cam, setCam] = useState('')
  const [rois, setRois] = useState([])
  const [draft, setDraft] = useState(emptyDraft())
  const [nat, setNat] = useState({ w: 960, h: 540 })   // do phan giai goc
  const [snapTick, setSnapTick] = useState(0)          // refresh anh nen
  const [msg, setMsg] = useState('')
  const svgRef = useRef(null)

  useEffect(() => {
    api.getCameras()
      .then(list => {
        const arr = Array.isArray(list) ? list : []
        setCameras(arr)
        if (arr.length && !cam) setCam(arr[0].id)
      })
      .catch(() => {})
  }, [])

  const loadRois = (camId) => {
    if (!camId) return
    api.getRois(camId)
      .then(data => setRois(Array.isArray(data) ? data : []))
      .catch(() => setRois([]))
  }

  useEffect(() => {
    if (!cam) return
    setDraft(emptyDraft())
    loadRois(cam)
    const c = cameras.find(c => c.id === cam)
    if (c?.resolution_w && c?.resolution_h) setNat({ w: c.resolution_w, h: c.resolution_h })
  }, [cam])

  // Click len anh -> them dinh polygon (toa do goc camera)
  const onSvgClick = (e) => {
    const rect = svgRef.current.getBoundingClientRect()
    const x = Math.round((e.clientX - rect.left) / rect.width * nat.w)
    const y = Math.round((e.clientY - rect.top) / rect.height * nat.h)
    setDraft(d => ({ ...d, points: [...d.points, [x, y]] }))
  }

  const undoPoint = () => setDraft(d => ({ ...d, points: d.points.slice(0, -1) }))
  const clearDraft = () => setDraft(emptyDraft())

  const save = async () => {
    if (draft.points.length < 3) { setMsg('Cần ≥ 3 điểm để tạo vùng'); return }
    if (!draft.name.trim())      { setMsg('Nhập tên vùng'); return }
    const dir = DIRECTIONS[draft.allowedDirIdx].value
    const body = {
      camera_id: cam,
      name: draft.name.trim(),
      polygon: JSON.stringify(draft.points),
      kind: draft.kind,
      alert_types: draft.kind === 'alert_zone' ? (draft.alertTypes.join(',') || 'entry') : 'entry',
      allowed_classes: draft.kind === 'lane' && draft.allowedClasses.length
        ? JSON.stringify(draft.allowedClasses) : null,
      allowed_direction: draft.kind === 'lane' && dir ? JSON.stringify(dir) : null,
    }
    try {
      await api.createRoi(body)
      setMsg('Đã lưu — áp dụng vào pipeline ngay (không cần restart)')
      clearDraft()
      loadRois(cam)
    } catch (err) {
      setMsg('Lỗi lưu: ' + err.message)
    }
  }

  const remove = async (id) => {
    await api.deleteRoi(id).catch(() => {})
    loadRois(cam)
    setMsg('Đã xóa vùng và cập nhật pipeline')
  }

  const toggleClass = (c) => setDraft(d => ({
    ...d,
    allowedClasses: d.allowedClasses.includes(c)
      ? d.allowedClasses.filter(x => x !== c)
      : [...d.allowedClasses, c],
  }))
  const toggleAlertType = (t) => setDraft(d => ({
    ...d,
    alertTypes: d.alertTypes.includes(t)
      ? d.alertTypes.filter(x => x !== t)
      : [...d.alertTypes, t],
  }))

  const parsePoly = (p) => { try { return JSON.parse(p) } catch { return [] } }
  const toPointsAttr = (pts) => pts.map(([x, y]) => `${x},${y}`).join(' ')

  return (
    <div className="min-h-screen bg-gray-950 p-6 text-white">
      <div className="flex items-center gap-4 mb-5">
        <button onClick={() => navigate('/')} className="text-gray-400 hover:text-white text-sm">
          ← Dashboard
        </button>
        <h1 className="text-xl font-bold">Cấu hình vùng quan tâm (ROI)</h1>
        <select
          value={cam}
          onChange={e => setCam(e.target.value)}
          className="ml-auto bg-gray-800 text-sm px-3 py-2 rounded border border-gray-700"
        >
          {cameras.map(c => <option key={c.id} value={c.id}>{c.id} {c.name && `— ${c.name}`}</option>)}
        </select>
        <button
          onClick={() => setSnapTick(t => t + 1)}
          className="bg-gray-800 hover:bg-gray-700 text-sm px-3 py-2 rounded border border-gray-700"
        >
          ⟳ Làm mới ảnh
        </button>
      </div>

      <div className="flex gap-5">
        {/* Canvas ve */}
        <div className="flex-1 min-w-0">
          <div className="relative inline-block w-full bg-black rounded-xl overflow-hidden border border-gray-800">
            <img
              key={`${cam}-${snapTick}`}
              src={cam ? api.snapshotUrl(cam) : ''}
              alt="snapshot"
              className="block w-full h-auto select-none"
              onLoad={e => setNat({ w: e.target.naturalWidth || nat.w, h: e.target.naturalHeight || nat.h })}
              draggable={false}
            />
            <svg
              ref={svgRef}
              onClick={onSvgClick}
              viewBox={`0 0 ${nat.w} ${nat.h}`}
              preserveAspectRatio="none"
              className="absolute inset-0 w-full h-full cursor-crosshair"
            >
              {/* ROI da luu */}
              {rois.filter(r => r.is_active).map(r => {
                const pts = parsePoly(r.polygon)
                const color = r.kind === 'lane' ? '#22c55e' : '#3b82f6'
                return (
                  <g key={r.id}>
                    <polygon points={toPointsAttr(pts)} fill={color} fillOpacity="0.18"
                             stroke={color} strokeWidth="2" />
                    {pts[0] && <text x={pts[0][0]} y={pts[0][1] - 4} fill={color}
                             fontSize={Math.max(12, nat.w / 60)}>{r.name}</text>}
                  </g>
                )
              })}
              {/* Draft dang ve */}
              {draft.points.length > 0 && (
                <polygon points={toPointsAttr(draft.points)}
                         fill="#f59e0b" fillOpacity="0.25" stroke="#f59e0b" strokeWidth="2" />
              )}
              {draft.points.map(([x, y], i) => (
                <circle key={i} cx={x} cy={y} r={Math.max(3, nat.w / 240)} fill="#f59e0b" />
              ))}
            </svg>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Click lên ảnh để thêm đỉnh. Xanh dương = vùng cảnh báo, xanh lá = vùng làn,
            cam = vùng đang vẽ. Độ phân giải gốc: {nat.w}×{nat.h}.
          </p>
        </div>

        {/* Bang dieu khien */}
        <div className="w-[340px] shrink-0 space-y-4">
          <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 space-y-3">
            <div className="text-sm font-bold text-gray-300">Vùng mới</div>
            <input
              value={draft.name}
              onChange={e => setDraft(d => ({ ...d, name: e.target.value }))}
              placeholder="Tên vùng (vd: lan_oto_trai)"
              className="w-full bg-gray-800 text-sm px-3 py-2 rounded border border-gray-700"
            />
            <div className="flex gap-2 text-sm">
              {['lane', 'alert_zone'].map(k => (
                <button key={k}
                  onClick={() => setDraft(d => ({ ...d, kind: k }))}
                  className={`flex-1 px-2 py-1.5 rounded border ${
                    draft.kind === k ? 'bg-indigo-600 border-indigo-500' : 'bg-gray-800 border-gray-700'}`}>
                  {k === 'lane' ? 'Làn (sai làn)' : 'Vùng cảnh báo'}
                </button>
              ))}
            </div>

            {draft.kind === 'lane' && (
              <>
                <div>
                  <div className="text-xs text-gray-400 mb-1">Lớp phương tiện được phép</div>
                  <div className="flex flex-wrap gap-1.5">
                    {VEHICLE_CLASSES.map(c => (
                      <button key={c} onClick={() => toggleClass(c)}
                        className={`text-xs px-2 py-1 rounded border ${
                          draft.allowedClasses.includes(c)
                            ? 'bg-green-700 border-green-600' : 'bg-gray-800 border-gray-700'}`}>
                        {c}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-400 mb-1">Hướng di chuyển hợp lệ</div>
                  <select
                    value={draft.allowedDirIdx}
                    onChange={e => setDraft(d => ({ ...d, allowedDirIdx: Number(e.target.value) }))}
                    className="w-full bg-gray-800 text-sm px-3 py-2 rounded border border-gray-700"
                  >
                    {DIRECTIONS.map((d, i) => <option key={i} value={i}>{d.label}</option>)}
                  </select>
                </div>
              </>
            )}

            {draft.kind === 'alert_zone' && (
              <div>
                <div className="text-xs text-gray-400 mb-1">Loại cảnh báo</div>
                <div className="flex flex-wrap gap-1.5">
                  {['entry', 'exit', 'loiter', 'speed'].map(t => (
                    <button key={t} onClick={() => toggleAlertType(t)}
                      className={`text-xs px-2 py-1 rounded border ${
                        draft.alertTypes.includes(t)
                          ? 'bg-blue-700 border-blue-600' : 'bg-gray-800 border-gray-700'}`}>
                      {t}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="text-xs text-gray-500">Điểm: {draft.points.length}</div>
            <div className="flex gap-2">
              <button onClick={save}
                className="flex-1 bg-emerald-600 hover:bg-emerald-700 text-sm px-3 py-2 rounded">
                Lưu vùng
              </button>
              <button onClick={undoPoint}
                className="bg-gray-800 hover:bg-gray-700 text-sm px-3 py-2 rounded border border-gray-700">
                ↶ Bỏ điểm
              </button>
              <button onClick={clearDraft}
                className="bg-gray-800 hover:bg-gray-700 text-sm px-3 py-2 rounded border border-gray-700">
                Hủy
              </button>
            </div>
            {msg && <div className="text-xs text-amber-400">{msg}</div>}
          </div>

          {/* Danh sach ROI hien co */}
          <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
            <div className="text-sm font-bold text-gray-300 mb-2">
              Vùng hiện có ({rois.length})
            </div>
            <div className="space-y-1.5 max-h-[40vh] overflow-auto">
              {rois.length === 0 && <div className="text-xs text-gray-600">Chưa có vùng nào</div>}
              {rois.map(r => (
                <div key={r.id} className="flex items-center gap-2 text-sm bg-gray-800/60 rounded px-2 py-1.5">
                  <span className={`w-2 h-2 rounded-full ${r.kind === 'lane' ? 'bg-green-500' : 'bg-blue-500'}`} />
                  <span className="flex-1 truncate">{r.name}</span>
                  <span className="text-[10px] text-gray-500">{r.kind}</span>
                  <button onClick={() => remove(r.id)}
                    className="text-xs text-red-400 hover:text-red-300">Xóa</button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
