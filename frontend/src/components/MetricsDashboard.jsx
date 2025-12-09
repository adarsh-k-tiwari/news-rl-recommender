import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { TrendingUp, MousePointerClick, Heart, X, BarChart3, Brain, Award, Zap } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { api } from '../services/api'

const MetricsDashboard = ({ userId, section = 'side' }) => {
  const [metrics, setMetrics] = useState({
    clicks: 0,
    impressions: 0,
    ctr: 0,
    likes: 0,
    dislikes: 0,
    history_length: 0,
    category_distribution: {},
    interactions_over_time: [],
    total_reward: 0.0,
    avg_reward: 0.0,
    recent_rewards: []
  })
  const [modelStats, setModelStats] = useState({
    current_model: 'Unknown',
    total_requests: 0,
    total_rewards: 0,
    avg_score: 0,
    avg_q_value: 0
  })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const data = await api.getMetrics(userId)
        if (data) {
          setMetrics(data)
        }
        setLoading(false)
      } catch (error) {
        console.error('Error fetching metrics:', error)
        setLoading(false)
      }
    }

    const fetchModelStats = async () => {
      try {
        const response = await fetch('/api/model-stats')
        const data = await response.json()
        setModelStats(data)
      } catch (error) {
        console.error('Error fetching model stats:', error)
      }
    }

    fetchMetrics()
    fetchModelStats()
    const interval = setInterval(() => {
      fetchMetrics()
      fetchModelStats()
    }, 2000) // Update every 2 seconds
    return () => clearInterval(interval)
  }, [userId])

  const categoryData = Object.entries(metrics.category_distribution).map(([name, value]) => ({
    name,
    value
  }))

  const COLORS = ['#7f1d1d', '#991b1b', '#b91c1c', '#dc2626', '#ef4444', '#f87171']

  if (loading) {
    return (
      <div className="p-6" style={{ background: 'rgba(255, 255, 255, 0.14)', borderRadius: '22px', boxShadow: '0 22px 45px rgba(70, 0, 0, 0.20)' }}>
        <div style={{ color: '#500000' }}>Loading metrics...</div>
      </div>
    )
  }

  // Render side metrics (RL Performance first, then Performance Metrics, then Category charts)
  if (section === 'side') {
    return (
      <div className="space-y-4">
      {/* RL Performance Metrics - First */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="p-6"
        style={{ background: 'rgba(255, 255, 255, 0.14)', borderRadius: '22px', border: '1px solid rgba(255, 255, 255, 0.35)', boxShadow: '0 22px 45px rgba(70, 0, 0, 0.20)', backdropFilter: 'blur(18px) saturate(125%)', borderTop: '4px solid #500000' }}
      >
        <h3 className="text-xl font-bold mb-4 flex items-center gap-2" style={{ color: '#500000' }}>
          <Zap size={24} />
          RL Performance Metrics
        </h3>
        <div className="space-y-3">
          <MetricCard
            icon={<Award size={20} />}
            label="Total Reward"
            value={metrics.total_reward || 0}
            color="text-yellow-300"
          />
          <MetricCard
            icon={<TrendingUp size={20} />}
            label="Avg Reward"
            value={`${metrics.avg_reward?.toFixed(2) || 0}`}
            color="text-green-300"
          />
          {metrics.prediction_stats && metrics.prediction_stats.total_predictions > 0 && (
            <div className="rounded-lg p-3" style={{ background: 'rgba(255, 255, 255, 0.95)', boxShadow: '0 4px 12px rgba(70, 0, 0, 0.1)' }}>
              <div className="text-sm font-semibold mb-2" style={{ color: '#500000' }}>Prediction Accuracy</div>
              <div className="text-2xl font-bold mb-2" style={{ color: '#500000' }}>
                {metrics.prediction_stats.accuracy}%
              </div>
              <div className="text-xs space-y-1" style={{ color: '#2b0a0a' }}>
                <div>✓ Correct: {metrics.prediction_stats.correct}</div>
                <div>✗ Wrong: {metrics.prediction_stats.wrong}</div>
                <div>⚠ Missed: {metrics.prediction_stats.missed_opportunities}</div>
                <div>✓ Avoided: {metrics.prediction_stats.correct_avoidances}</div>
              </div>
            </div>
          )}
          <div className="rounded-lg p-3 mt-3" style={{ background: 'rgba(255, 255, 255, 0.95)', boxShadow: '0 4px 12px rgba(70, 0, 0, 0.1)' }}>
            <div className="flex items-center gap-2 mb-2">
              <Brain size={18} style={{ color: '#500000' }} />
              <span className="text-sm font-medium" style={{ color: '#500000' }}>Model: {modelStats.current_model.includes('DQN') ? 'CMAB' : modelStats.current_model}</span>
            </div>
            <div className="text-xs space-y-1" style={{ color: '#2b0a0a' }}>
              <div>Requests: {modelStats.total_requests}</div>
              <div>Avg Q-Value: {modelStats.avg_q_value.toFixed(4)}</div>
              <div>Avg Score: {modelStats.avg_score.toFixed(4)}</div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Performance Metrics - Second */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="p-6"
        style={{ background: 'rgba(255, 255, 255, 0.14)', borderRadius: '22px', border: '1px solid rgba(255, 255, 255, 0.35)', boxShadow: '0 22px 45px rgba(70, 0, 0, 0.20)', backdropFilter: 'blur(18px) saturate(125%)', borderTop: '4px solid #500000' }}
      >
        <h3 className="text-xl font-bold mb-4 flex items-center gap-2" style={{ color: '#500000' }}>
          <BarChart3 size={24} />
          Performance Metrics
        </h3>
        
        <div className="grid grid-cols-2 gap-4">
          <MetricCard
            icon={<MousePointerClick size={20} />}
            label="CTR"
            value={`${metrics.ctr}%`}
            color="text-blue-300"
          />
          <MetricCard
            icon={<TrendingUp size={20} />}
            label="Clicks"
            value={metrics.clicks}
            color="text-green-300"
          />
          <MetricCard
            icon={<Heart size={20} />}
            label="Likes"
            value={metrics.likes}
            color="text-pink-300"
          />
          <MetricCard
            icon={<X size={20} />}
            label="Dislikes"
            value={metrics.dislikes}
            color="text-red-300"
          />
        </div>
      </motion.div>

      {/* Category Preferences - Third */}
      {categoryData.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="p-6"
          style={{ background: 'rgba(255, 255, 255, 0.14)', borderRadius: '22px', border: '1px solid rgba(255, 255, 255, 0.35)', boxShadow: '0 22px 45px rgba(70, 0, 0, 0.20)', backdropFilter: 'blur(18px) saturate(125%)', borderTop: '4px solid #500000' }}
        >
          <h3 className="text-xl font-bold mb-4" style={{ color: '#500000' }}>Category Preferences</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
              <Pie
                data={categoryData}
                cx="50%"
                cy="50%"
                labelLine={true}
                label={({ name, percent, x, y, midAngle, innerRadius, outerRadius }) => {
                  const RADIAN = Math.PI / 180;
                  const radius = outerRadius + 25;
                  const x1 = x + radius * Math.cos(-midAngle * RADIAN);
                  const y1 = y + radius * Math.sin(-midAngle * RADIAN);
                  const shortName = name.length > 8 ? name.substring(0, 8) : name;
                  return (
                    <text 
                      x={x1} 
                      y={y1} 
                      fill="#2b0a0a" 
                      textAnchor={x1 > x ? 'start' : 'end'} 
                      dominantBaseline="central"
                      fontSize={11}
                      fontWeight={500}
                    >
                      {`${shortName} ${(percent * 100).toFixed(0)}%`}
                    </text>
                  );
                }}
                outerRadius={60}
                fill="#8884d8"
                dataKey="value"
              >
                {categoryData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </motion.div>
      )}


    </div>
    )
  }

  // Render left sidebar metrics (Session Stats only)
  if (section === 'left') {
    return (
      <div className="space-y-4">
        {/* Session Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-6"
          style={{ background: 'rgba(255, 255, 255, 0.14)', borderRadius: '22px', border: '1px solid rgba(255, 255, 255, 0.35)', boxShadow: '0 22px 45px rgba(70, 0, 0, 0.20)', backdropFilter: 'blur(18px) saturate(125%)', borderTop: '4px solid #500000' }}
        >
          <h3 className="text-xl font-bold mb-4" style={{ color: '#500000' }}>Session Statistics</h3>
          <div className="space-y-2" style={{ color: '#2b0a0a' }}>
            <div className="flex justify-between">
              <span>Total Impressions:</span>
              <span className="font-semibold">{metrics.impressions}</span>
            </div>
            <div className="flex justify-between">
              <span>History Length:</span>
              <span className="font-semibold">{metrics.history_length}</span>
            </div>
            <div className="flex justify-between">
              <span>Engagement Rate:</span>
              <span className="font-semibold">
                {metrics.impressions > 0 
                  ? ((metrics.clicks / metrics.impressions) * 100).toFixed(1)
                  : 0}%
              </span>
            </div>
          </div>
        </motion.div>

        {/* Category Bar Chart - Under Session Stats */}
        {categoryData.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="p-6"
            style={{ background: 'rgba(255, 255, 255, 0.14)', borderRadius: '22px', border: '1px solid rgba(255, 255, 255, 0.35)', boxShadow: '0 22px 45px rgba(70, 0, 0, 0.20)', backdropFilter: 'blur(18px) saturate(125%)', borderTop: '4px solid #500000' }}
          >
            <h3 className="text-xl font-bold mb-4" style={{ color: '#500000' }}>Category Distribution</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={categoryData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.2)" />
                <XAxis 
                  dataKey="name" 
                  tick={{ fill: '#2b0a0a', fontSize: 12 }}
                  angle={-45}
                  textAnchor="end"
                  height={80}
                />
                <YAxis tick={{ fill: '#2b0a0a', fontSize: 12 }} />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'rgba(0,0,0,0.8)', 
                    border: 'none',
                    borderRadius: '8px',
                    color: 'white'
                  }}
                />
                <Bar dataKey="value" fill="#500000" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </motion.div>
        )}
      </div>
    )
  }

  // Render bottom metrics (Recent Rewards only)
  return (
    <div className="space-y-4">
      {/* Reward History Chart */}
      {metrics.recent_rewards && metrics.recent_rewards.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="p-6"
          style={{ background: 'rgba(255, 255, 255, 0.14)', borderRadius: '22px', border: '1px solid rgba(255, 255, 255, 0.35)', boxShadow: '0 22px 45px rgba(70, 0, 0, 0.20)', backdropFilter: 'blur(18px) saturate(125%)', borderTop: '4px solid #500000' }}
        >
          <h3 className="text-xl font-bold mb-4" style={{ color: '#500000' }}>Recent Rewards</h3>
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={metrics.recent_rewards.map((r, i) => ({ index: i, reward: r }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.2)" />
              <XAxis 
                dataKey="index" 
                tick={{ fill: '#2b0a0a', fontSize: 10 }}
              />
              <YAxis 
                domain={[0, 1]}
                tick={{ fill: '#2b0a0a', fontSize: 10 }}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'rgba(0,0,0,0.8)', 
                  border: 'none',
                  borderRadius: '8px',
                  color: 'white'
                }}
              />
              <Bar 
                dataKey="reward" 
                radius={[4, 4, 0, 0]}
                fill="#500000"
              />
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      )}
    </div>
  )
}

const MetricCard = ({ icon, label, value, color }) => (
  <div className="rounded-xl p-4" style={{ background: 'rgba(255, 255, 255, 0.95)', boxShadow: '0 10px 22px rgba(70, 0, 0, 0.15)', borderRadius: '18px' }}>
    <div className={`flex items-center gap-2 mb-2 ${color}`}>
      {icon}
      <span className="text-sm font-medium" style={{ color: '#2b0a0a' }}>{label}</span>
    </div>
    <div className="text-2xl font-bold" style={{ color: '#500000' }}>{value}</div>
  </div>
)

export default MetricsDashboard

