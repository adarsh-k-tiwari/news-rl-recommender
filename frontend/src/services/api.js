import axios from 'axios'

const API_BASE_URL = '/api'

export const api = {
  async getModels() {
    const response = await axios.get(`${API_BASE_URL}/models`)
    return response.data
  },

  async setModel(modelName) {
    const response = await axios.post(`${API_BASE_URL}/models/${modelName}`)
    return response.data
  },

  async getRecommendations(userId, history = [], categories = [], count = 10) {
    const response = await axios.post(`${API_BASE_URL}/recommendations`, {
      user_id: userId,
      history,
      categories,
      count
    })
    return response.data
  },

  async recordInteraction(userId, newsId, action) {
    const response = await axios.post(`${API_BASE_URL}/interaction`, {
      user_id: userId,
      news_id: newsId,
      action
    })
    return response.data
  },

  async getMetrics(userId) {
    const response = await axios.get(`${API_BASE_URL}/metrics/${userId}`)
    return response.data
  },

  async getCategories() {
    const response = await axios.get(`${API_BASE_URL}/categories`)
    return response.data
  },

  async getArticle(newsId) {
    const response = await axios.get(`${API_BASE_URL}/article/${newsId}`)
    return response.data
  },

  async getModelStats() {
    const response = await axios.get(`${API_BASE_URL}/model-stats`)
    return response.data
  },

  async validateLearning(userId) {
    const response = await axios.get(`${API_BASE_URL}/validate-learning/${userId}`)
    return response.data
  }
}

