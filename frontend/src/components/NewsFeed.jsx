import React, { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Heart, X, ChevronDown } from 'lucide-react'
import { api } from '../services/api'

const NewsFeed = ({ userId, categories = [] }) => {
  const [articles, setArticles] = useState([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(false)
  const [lastFeedback, setLastFeedback] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [dragY, setDragY] = useState(0)
  const touchStartY = useRef(0)
  const containerRef = useRef(null)

  // Load user's session history from backend on mount
  useEffect(() => {
    const loadSessionHistory = async () => {
      try {
        const metrics = await api.getMetrics(userId)
        if (metrics && metrics.history_length > 0) {
          // Backend session has history - sync it
          console.log('Syncing history from backend session:', metrics.likes, 'likes')
          setHistory(metrics.interactions_over_time
            .filter(i => i.action === 'like')
            .map(i => i.news_id))
        }
      } catch (error) {
        console.log('Could not load session history:', error)
      }
    }
    loadSessionHistory()
  }, [userId])

  const loadRecommendations = useCallback(async () => {
    if (loading) return
    setLoading(true)
    try {
      const data = await api.getRecommendations(userId, history, categories, 20)
      if (data && data.recommendations) {
        setArticles(prev => [...prev, ...data.recommendations])
        
        // Log debug info to verify personalization
        if (data.debug) {
          console.log('Recommendation debug:', data.debug)
          if (data.debug.user_likes_count > 20) {
            console.log('✅ Personalized recommendations based on', data.debug.user_likes_count, 'likes')
            console.log('User prefers:', data.debug.user_top_categories)
            console.log('Recommended:', data.debug.recommended_categories)
          }
        }
      }
    } catch (error) {
      console.error('Error loading recommendations:', error)
      // Show error message to user
      alert('Failed to load recommendations. Make sure the backend API is running on port 5000.')
    } finally {
      setLoading(false)
    }
  }, [userId, history, categories])

  useEffect(() => {
    loadRecommendations()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleLike = async (newsId) => {
    // Prevent double-clicks and ensure we have articles
    if (loading || articles.length === 0) {
      console.log('Cannot like: loading =', loading, 'articles =', articles.length)
      return
    }
    
    console.log('Like clicked, current index:', currentIndex, 'articles length:', articles.length, 'current article:', articles[currentIndex]?.news_id)
    
    // Update history
    const newHistory = [...history, newsId]
    setHistory(newHistory)
    
    // Move to next article immediately
    setCurrentIndex(prev => {
      const currentArticlesLength = articles.length
      if (currentArticlesLength === 0) return 0
      
      let nextIndex = prev + 1
      console.log('Moving to next index:', nextIndex, 'from', prev, 'articles:', currentArticlesLength)
      
      // If we're at or past the end, stay at the last article
      if (nextIndex >= currentArticlesLength) {
        nextIndex = currentArticlesLength - 1
        console.log('At end, staying at index:', nextIndex)
        
        // Load more articles immediately
        console.log('Loading more articles immediately...')
        api.getRecommendations(userId, newHistory, categories, 20)
          .then(data => {
            if (data && data.recommendations && data.recommendations.length > 0) {
              console.log('Loaded', data.recommendations.length, 'new articles')
              setArticles(prevArticles => {
                const updated = [...prevArticles, ...data.recommendations]
                // Move to the first new article
                setCurrentIndex(prevArticles.length)
                return updated
              })
            }
          })
          .catch(err => console.error('Error loading more:', err))
      } else {
        // Check if we need to preload more
        if (nextIndex >= currentArticlesLength - 3) {
          console.log('Preloading more articles...')
          api.getRecommendations(userId, newHistory, categories, 20)
            .then(data => {
              if (data && data.recommendations) {
                setArticles(prevArticles => [...prevArticles, ...data.recommendations])
              }
            })
            .catch(err => console.error('Error preloading:', err))
        }
      }
      
      return nextIndex
    })
    
    // Record interaction and show feedback (reward, model)
    api.recordInteraction(userId, newsId, 'like')
      .then(res => {
        if (res) {
          setLastFeedback({ reward: res.reward, total: res.total_reward, model: res.model })
          setTimeout(() => setLastFeedback(null), 3000)
        }
      })
      .catch(err => console.error('Error recording like:', err))
  }

  const handleDislike = async (newsId) => {
    // Prevent double-clicks and ensure we have articles
    if (loading || articles.length === 0) {
      console.log('Cannot dislike: loading =', loading, 'articles =', articles.length)
      return
    }
    
    console.log('Dislike clicked, current index:', currentIndex, 'articles length:', articles.length, 'current article:', articles[currentIndex]?.news_id)
    
    // Move to next article immediately
    setCurrentIndex(prev => {
      const currentArticlesLength = articles.length
      if (currentArticlesLength === 0) return 0
      
      let nextIndex = prev + 1
      console.log('Moving to next index:', nextIndex, 'from', prev, 'articles:', currentArticlesLength)
      
      // If we're at or past the end, stay at the last article
      if (nextIndex >= currentArticlesLength) {
        nextIndex = currentArticlesLength - 1
        console.log('At end, staying at index:', nextIndex)
        
        // Load more articles immediately
        console.log('Loading more articles immediately...')
        api.getRecommendations(userId, history, categories, 20)
          .then(data => {
            if (data && data.recommendations && data.recommendations.length > 0) {
              console.log('Loaded', data.recommendations.length, 'new articles')
              setArticles(prevArticles => {
                const updated = [...prevArticles, ...data.recommendations]
                // Move to the first new article
                setCurrentIndex(prevArticles.length)
                return updated
              })
            }
          })
          .catch(err => console.error('Error loading more:', err))
      } else {
        // Check if we need to preload more
        if (nextIndex >= currentArticlesLength - 3) {
          console.log('Preloading more articles...')
          api.getRecommendations(userId, history, categories, 20)
            .then(data => {
              if (data && data.recommendations) {
                setArticles(prevArticles => [...prevArticles, ...data.recommendations])
              }
            })
            .catch(err => console.error('Error preloading:', err))
        }
      }
      
      return nextIndex
    })
    
    // Record interaction and show feedback (reward, model)
    api.recordInteraction(userId, newsId, 'dislike')
      .then(res => {
        if (res) {
          setLastFeedback({ reward: res.reward, total: res.total_reward, model: res.model })
          setTimeout(() => setLastFeedback(null), 3000)
        }
      })
      .catch(err => console.error('Error recording dislike:', err))
  }

  const handleTouchStart = (e) => {
    touchStartY.current = e.touches[0].clientY
    setIsDragging(true)
  }

  const handleTouchMove = (e) => {
    if (!isDragging) return
    const currentY = e.touches[0].clientY
    const diff = currentY - touchStartY.current
    setDragY(diff)
  }

  const handleTouchEnd = () => {
    if (Math.abs(dragY) > 100) {
      if (dragY > 0 && currentIndex > 0) {
        // Swipe down - go to previous
        setCurrentIndex(prev => prev - 1)
      } else if (dragY < 0 && currentIndex < articles.length - 1) {
        // Swipe up - go to next
        setCurrentIndex(prev => prev + 1)
      }
    }
    setIsDragging(false)
    setDragY(0)
  }

  const handleWheel = useCallback((e) => {
    if (Math.abs(e.deltaY) > 50) {
      if (e.deltaY > 0 && currentIndex < articles.length - 1) {
        setCurrentIndex(prev => prev + 1)
      } else if (e.deltaY < 0 && currentIndex > 0) {
        setCurrentIndex(prev => prev - 1)
      }
    }
  }, [currentIndex, articles.length])

  useEffect(() => {
    const container = containerRef.current
    if (container) {
      container.addEventListener('wheel', handleWheel, { passive: false })
      return () => container.removeEventListener('wheel', handleWheel)
    }
  }, [handleWheel])

  // Ensure index is within bounds and update if needed
  useEffect(() => {
    if (articles.length > 0) {
      if (currentIndex >= articles.length) {
        setCurrentIndex(Math.max(0, articles.length - 1))
      } else if (currentIndex < 0) {
        setCurrentIndex(0)
      }
    }
  }, [articles.length, currentIndex])
  
  // Calculate safe index for rendering
  const safeIndex = articles.length > 0 
    ? Math.min(Math.max(0, currentIndex), articles.length - 1)
    : 0
  const currentArticle = articles[safeIndex]

  if (!currentArticle && !loading && articles.length === 0) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="text-white text-xl mb-4">Loading articles...</div>
          <div className="text-white/60 text-sm">
            {articles.length === 0 && 'Make sure the backend API is running on port 5000'}
          </div>
        </div>
      </div>
    )
  }

  if (!currentArticle && articles.length > 0) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-white text-xl">No more articles available</div>
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className="relative overflow-hidden"
      style={{ minHeight: '600px' }}
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
    >
      <AnimatePresence mode="wait" initial={false}>
        {currentArticle && (
          <motion.div
            key={`article-${safeIndex}-${currentArticle.news_id}`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ 
              opacity: 1, 
              y: dragY,
              scale: isDragging ? 0.98 : 1
            }}
            exit={{ opacity: 0, y: -20, scale: 0.95 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className="w-full"
          >
            <div className="w-full overflow-hidden" style={{ background: 'rgba(255, 255, 255, 0.95)', borderRadius: '22px', boxShadow: '0 22px 45px rgba(70, 0, 0, 0.20)', borderTop: '4px solid #500000' }}>
              {/* Article Content */}
              <div className="p-8">
                <div className="mb-4">
                  <span className="inline-block px-3 py-1 rounded-full text-sm font-semibold" style={{ background: 'rgba(115, 0, 0, 0.08)', color: '#500000', border: '1px solid rgba(255, 255, 255, 0.25)', borderRadius: '18px' }}>
                    {currentArticle.category}
                  </span>
                </div>
                
                <h2 className="text-3xl font-bold mb-4" style={{ color: '#500000' }}>
                  {currentArticle.title}
                </h2>
                
                {currentArticle.abstract && (
                  <p className="text-gray-600 text-lg leading-relaxed mb-6">
                    {currentArticle.abstract}
                  </p>
                )}

                {/* Progress Indicator */}
                <div className="mb-6">
                  <div className="flex items-center gap-2 text-sm text-gray-500">
                    <span>Article {safeIndex + 1} of {articles.length}</span>
                    <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full transition-all duration-300"
                        style={{ background: '#500000' }}
                        style={{ width: `${((safeIndex + 1) / articles.length) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="px-8 py-6 flex justify-center gap-6" style={{ background: '#f7f1eb' }}>
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => handleDislike(currentArticle.news_id)}
                  className="p-4 bg-gray-200 rounded-full text-gray-600 hover:bg-gray-300 transition-colors"
                >
                  <X size={32} />
                </motion.button>

                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => handleLike(currentArticle.news_id)}
                  className="p-4 rounded-full transition-colors"
                  style={{ background: '#500000', color: '#ffffff', boxShadow: '0 4px 12px rgba(70, 0, 0, 0.1)' }}
                >
                  <Heart size={32} />
                </motion.button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loading Indicator */}
      {loading && (
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2">
          <div className="bg-white/90 backdrop-blur-sm px-4 py-2 rounded-full text-sm text-gray-700">
            Loading more articles...
          </div>
        </div>
      )}

      {/* Scroll Hint */}
      {currentIndex === 0 && articles.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="absolute bottom-8 left-1/2 transform -translate-x-1/2 text-white text-center"
        >
          <ChevronDown size={24} className="mx-auto mb-2 animate-bounce" />
          <p className="text-sm">Scroll or swipe to see more</p>
        </motion.div>
      )}
    </div>
  )

  // Floating feedback box for recent interaction
  return (
    <>
      {lastFeedback && (
        <div className="fixed bottom-6 right-6 bg-white/95 text-gray-900 rounded-lg shadow-lg px-4 py-2 z-50">
          <div className="text-sm font-semibold">Feedback</div>
          <div className="text-sm">Reward: {lastFeedback.reward} • Total: {lastFeedback.total}</div>
          <div className="text-xs text-gray-500">Model: {lastFeedback.model.includes('DQN') ? 'CMAB' : lastFeedback.model}</div>
        </div>
      )}
      <div
        ref={containerRef}
        className="relative h-screen overflow-hidden"
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
      >
        <AnimatePresence mode="wait" initial={false}>
          {currentArticle && (
            <motion.div
              key={`article-${safeIndex}-${currentArticle.news_id}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ 
                opacity: 1, 
                y: dragY,
                scale: isDragging ? 0.98 : 1
              }}
              exit={{ opacity: 0, y: -20, scale: 0.95 }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
              className="absolute inset-0 flex items-center justify-center p-4"
            >
              <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full overflow-hidden">
                {/* Article Content */}
                <div className="p-8">
                  <div className="mb-4">
                    <span className="inline-block px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm font-semibold">
                      {currentArticle.category}
                    </span>
                  </div>
                  
                  <h2 className="text-3xl font-bold text-gray-900 mb-4">
                    {currentArticle.title}
                  </h2>
                  
                  {currentArticle.abstract && (
                    <p className="text-gray-600 text-lg leading-relaxed mb-6">
                      {currentArticle.abstract}
                    </p>
                  )}

                  {/* Progress Indicator */}
                  <div className="mb-6">
                    <div className="flex items-center gap-2 text-sm text-gray-500">
                      <span>Article {safeIndex + 1} of {articles.length}</span>
                      <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-purple-500 to-blue-500 transition-all duration-300"
                          style={{ width: `${((safeIndex + 1) / articles.length) * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="bg-gray-50 px-8 py-6 flex justify-center gap-6">
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => handleDislike(currentArticle.news_id)}
                    className="p-4 bg-red-100 rounded-full text-red-600 hover:bg-red-200 transition-colors"
                  >
                    <X size={32} />
                  </motion.button>

                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => handleLike(currentArticle.news_id)}
                    className="p-4 bg-green-100 rounded-full text-green-600 hover:bg-green-200 transition-colors"
                  >
                    <Heart size={32} />
                  </motion.button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Loading Indicator */}
        {loading && (
          <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2">
            <div className="bg-white/90 backdrop-blur-sm px-4 py-2 rounded-full text-sm text-gray-700">
              Loading more articles...
            </div>
          </div>
        )}

        {/* Scroll Hint */}
        {currentIndex === 0 && articles.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute bottom-8 left-1/2 transform -translate-x-1/2 text-white text-center"
          >
            <ChevronDown size={24} className="mx-auto mb-2 animate-bounce" />
            <p className="text-sm">Scroll or swipe to see more</p>
          </motion.div>
        )}
      </div>
    </>
  )
}

export default NewsFeed

