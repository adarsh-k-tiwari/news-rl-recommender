import React, { useState } from 'react'
import { api } from '../services/api'

const SimulationPanel = ({ userId, onSimulationComplete, onSimulationStart }) => {
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [totalInteractions, setTotalInteractions] = useState(1000)
  const [preferredGenres, setPreferredGenres] = useState('politics,sports')
  const [matchProb, setMatchProb] = useState(0.9)
  const [otherProb, setOtherProb] = useState(0.02)
  const [results, setResults] = useState(null)
  const [validationResults, setValidationResults] = useState(null)

  const startSimulation = async () => {
    setIsRunning(true)
    setProgress(0)
    setResults(null)
    
    // Notify parent to clear/reset feed
    if (onSimulationStart) {
      onSimulationStart()
    }

    const genres = preferredGenres.split(',').map(g => g.trim().toLowerCase())
    let history = []
    let interactionsDone = 0
    const total = totalInteractions
    let lastDebugInfo = null

    try {
      while (interactionsDone < total) {
        // Request recommendations
        const data = await api.getRecommendations(userId, history, genres, 10)
        const recs = data.recommendations || []
        
        // Store last debug info to show at the end
        if (data.debug) {
          lastDebugInfo = data.debug
        }

        if (!recs.length) break

        for (const art of recs) {
          if (interactionsDone >= total) break

          const category = (art.category || '').toLowerCase()
          const prob = genres.includes(category) ? matchProb : otherProb
          const liked = Math.random() < prob
          const action = liked ? 'like' : 'dislike'

          // Send interaction
          await api.recordInteraction(userId, art.news_id, action)

          interactionsDone++
          if (liked) history.push(art.news_id)

          setProgress(Math.round((interactionsDone / total) * 100))

          // Small delay to avoid overwhelming the server
          await new Promise(resolve => setTimeout(resolve, 50))
        }
      }

      // Get final metrics
      const metrics = await api.getMetrics(userId)
      if (lastDebugInfo) {
        metrics.debug = lastDebugInfo
      }
      setResults(metrics)
      if (onSimulationComplete) onSimulationComplete(metrics)
    } catch (error) {
      console.error('Simulation error:', error)
      alert('Simulation failed: ' + error.message)
    } finally {
      setIsRunning(false)
    }
  }

  const stopSimulation = () => {
    setIsRunning(false)
  }

  const validateLearning = async () => {
    try {
      const validation = await api.validateLearning(userId)
      setValidationResults(validation)
    } catch (error) {
      console.error('Validation error:', error)
      alert('Validation failed: ' + error.message)
    }
  }

  return (
    <div className="p-6" style={{ background: 'rgba(255, 255, 255, 0.14)', borderRadius: '22px', border: '1px solid rgba(255, 255, 255, 0.35)', boxShadow: '0 22px 45px rgba(70, 0, 0, 0.20)', backdropFilter: 'blur(18px) saturate(125%)' }}>
      <h2 className="text-xl font-bold mb-4" style={{ color: '#500000' }}>Simulation Panel</h2>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1" style={{ color: '#2b0a0a' }}>Total Interactions</label>
          <input
            type="number"
            value={totalInteractions}
            onChange={(e) => setTotalInteractions(parseInt(e.target.value))}
            className="w-full p-2 border rounded"
            disabled={isRunning}
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1" style={{ color: '#2b0a0a' }}>Preferred Genres (comma-separated)</label>
          <input
            type="text"
            value={preferredGenres}
            onChange={(e) => setPreferredGenres(e.target.value)}
            className="w-full p-2 border rounded"
            disabled={isRunning}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1" style={{ color: '#2b0a0a' }}>Match Prob</label>
            <input
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={matchProb}
              onChange={(e) => setMatchProb(parseFloat(e.target.value))}
              className="w-full p-2 border rounded"
              disabled={isRunning}
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1" style={{ color: '#2b0a0a' }}>Other Prob</label>
            <input
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={otherProb}
              onChange={(e) => setOtherProb(parseFloat(e.target.value))}
              className="w-full p-2 border rounded"
              disabled={isRunning}
            />
          </div>
        </div>

        <div className="flex gap-2">
          {!isRunning ? (
            <button
              onClick={startSimulation}
              className="flex-1 text-white py-2 px-4 rounded transition-colors"
              style={{ background: '#500000', borderRadius: '999px', boxShadow: '0 12px 25px rgba(80, 0, 0, 0.3)' }}
              onMouseEnter={(e) => e.target.style.background = '#730000'}
              onMouseLeave={(e) => e.target.style.background = '#500000'}
            >
              Start Simulation
            </button>
          ) : (
            <button
              onClick={stopSimulation}
              className="flex-1 bg-gray-600 text-white py-2 px-4 rounded hover:bg-gray-700 transition-colors"
            >
              Stop Simulation
            </button>
          )}
        </div>

        {isRunning && (
          <div className="text-center">
            <div className="text-sm mb-2" style={{ color: '#2b0a0a' }}>Progress: {progress}%</div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="h-2 rounded-full transition-all duration-300"
                style={{ background: '#500000', width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {results && (
          <div className="mt-4 p-4 rounded-xl" style={{ background: 'rgba(255, 255, 255, 0.95)', boxShadow: '0 10px 22px rgba(70, 0, 0, 0.15)', borderRadius: '18px' }}>
            <h3 className="font-semibold mb-2" style={{ color: '#500000' }}>Simulation Results</h3>
            <div className="text-sm space-y-1" style={{ color: '#2b0a0a' }}>
              <div>Clicks: {results.clicks}</div>
              <div>Impressions: {results.impressions}</div>
              <div>CTR: {results.ctr}%</div>
              <div>Likes: {results.likes}</div>
              <div>Dislikes: {results.dislikes}</div>
              <div>Total Reward: {results.total_reward}</div>
              <div>Avg Reward: {results.avg_reward}</div>
              
              {results.debug && (
                <>
                  <hr className="my-2" />
                  <div className="font-semibold">Model Learning:</div>
                  <div>User Liked: {results.debug.user_likes_count} articles</div>
                  <div>Top Categories: {JSON.stringify(results.debug.user_top_categories)}</div>
                  <div>Recommended: {JSON.stringify(results.debug.recommended_categories)}</div>
                </>
              )}
            </div>
            <button
              onClick={validateLearning}
              className="mt-2 w-full text-white py-2 px-4 text-sm transition-colors"
              style={{ background: '#500000', borderRadius: '999px', boxShadow: '0 12px 25px rgba(80, 0, 0, 0.3)' }}
              onMouseEnter={(e) => e.target.style.background = '#730000'}
              onMouseLeave={(e) => e.target.style.background = '#500000'}
            >
              Validate Model Learning
            </button>
          </div>
        )}

        {validationResults && (
          <div className="mt-4 p-4 rounded-xl" style={{ background: 'rgba(255, 255, 255, 0.95)', boxShadow: '0 10px 22px rgba(70, 0, 0, 0.15)', borderRadius: '18px', borderLeft: '4px solid #500000' }}>
            <h3 className="font-semibold mb-2" style={{ color: '#500000' }}>Learning Validation</h3>
            <div className="text-sm space-y-1">
              <div className="font-bold text-lg">
                Status: <span className={validationResults.learning_indicator === 'GOOD' ? 'text-green-600' : validationResults.learning_indicator === 'WEAK' ? 'text-yellow-600' : 'text-red-600'}>
                  {validationResults.learning_indicator}
                </span>
              </div>
              <div>Preferred Categories: {validationResults.preferred_categories.join(', ')}</div>
              <div>Preferred Avg Score: {validationResults.preferred_articles_scores.mean.toFixed(4)}</div>
              <div>Other Avg Score: {validationResults.other_articles_scores.mean.toFixed(4)}</div>
              <div>Score Difference: {validationResults.score_difference.mean_diff.toFixed(4)}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default SimulationPanel