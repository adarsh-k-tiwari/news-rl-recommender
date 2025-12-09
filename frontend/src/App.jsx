import React, { useState, useEffect } from 'react'
import NewsFeed from './components/NewsFeed'
import MetricsDashboard from './components/MetricsDashboard'
import ModelSelector from './components/ModelSelector'
import SimulationPanel from './components/SimulationPanel'
import { api } from './services/api'
function App() {
  const [userId] = useState(() => `user_${Date.now()}`)
  const [currentModel, setCurrentModel] = useState('supervised')
  const [availableModels, setAvailableModels] = useState([])
  const [categories, setCategories] = useState([])
  const [selectedCategories, setSelectedCategories] = useState([])
  const [apiError, setApiError] = useState(false)
  const [feedKey, setFeedKey] = useState(0) // Key to force NewsFeed reload
  useEffect(() => {
    api.getModels()
      .then(data => {
        setAvailableModels(data.models || [])
        if (data.models && data.models.length > 0) {
          setCurrentModel(data.models[0])
        }
        setApiError(false)
      })
      .catch(error => {
        console.error('Error loading models:', error)
        setAvailableModels(['supervised'])
        setApiError(true)
      })

    api.getCategories()
      .then(data => {
        setCategories(data.categories || [])
      })
      .catch(error => {
        console.error('Error loading categories:', error)
      })
  }, [])

  const handleModelChange = async (modelName) => {
    try {
      await api.setModel(modelName)
      setCurrentModel(modelName)
    } catch (error) {
      console.error('Error setting model:', error)
    }
  }

  return (
    <div className="min-h-screen" style={{ backgroundColor: '#f2ebe5' }}>
      <div className="container mx-auto px-4 py-6">
        {/* Header - A&M Style */}
        <div className="mb-6 text-white p-6 rounded-lg relative" style={{ background: 'linear-gradient(to right, #500000, #e7dadaff)', boxShadow: '0 10px 22px rgba(70, 0, 0, 0.15)', borderRadius: '22px' }}>
          {/* A&M Logo */}
          <img 
            src="/Artboard 2.svg" 
            alt="Texas A&M Logo" 
            className="absolute top-4 right-6"
            style={{ height: '90px', width: 'auto' }}
          />
          <h1 className="text-4xl font-bold mb-2" style={{ color: '#ffffff' }}>
            News RL Recommender
          </h1>
          <p style={{ color: 'rgba(255, 255, 255, 0.9)' }}>
            Reinforcement Learning Powered News Recommendation System
          </p>
        </div>

        {/* API Error Warning */}
        {apiError && (
          <div className="mb-6 bg-yellow-50 border-l-4 border-yellow-500 rounded-lg p-4">
            <p className="text-yellow-800 font-semibold">
              ⚠️ Backend API not connected. Please start the backend server:
            </p>
            <code className="text-yellow-700 text-sm mt-2 block bg-yellow-100 p-2 rounded">
              python api_server.py
            </code>
          </div>
        )}

        {/* Model Selector */}
        <div className="mb-6">
          <ModelSelector
            models={availableModels}
            currentModel={currentModel}
            onModelChange={handleModelChange}
          />
        </div>

        {/* Main Layout - Article centered with metrics wrapping */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Sidebar - Simulation Panel + Session Stats */}
          <div className="lg:col-span-3 space-y-6">
            <SimulationPanel 
              userId={userId}
              onSimulationComplete={() => {
                console.log('Simulation complete, reloading feed with trained model')
                setFeedKey(prev => prev + 1)
              }}
              onSimulationStart={() => {
                console.log('Simulation starting')
              }}
            />
            <MetricsDashboard userId={userId} section="left" />
          </div>
          <div className="lg:col-span-6 space-y-6">
            <NewsFeed
              key={feedKey}
              userId={userId}
              categories={selectedCategories}
            />
            <MetricsDashboard userId={userId} section="bottom" />
          </div>

          {/* Right Sidebar - RL Performance + Performance Metrics + Category Charts */}
          <div className="lg:col-span-3 space-y-6">
            <MetricsDashboard userId={userId} section="side" />
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
