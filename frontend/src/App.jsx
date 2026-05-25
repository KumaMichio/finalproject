import React from 'react'
import { Routes, Route } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import IncidentDetail from './pages/IncidentDetail'
import AlertManagement from './pages/AlertManagement'
import ObjectHistory from './pages/ObjectHistory'

export default function App() {
  return (
    <Routes>
      <Route path="/"                     element={<Dashboard />} />
      <Route path="/incident/:id"         element={<IncidentDetail />} />
      <Route path="/alerts"               element={<AlertManagement />} />
      <Route path="/object/:global_id"    element={<ObjectHistory />} />
    </Routes>
  )
}
