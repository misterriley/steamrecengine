<!-- components/UserRecommendations.vue -->
<template>
  <div>
    <h2>Recommendations</h2>
    <button
      @click="generateRecommendations"
      :disabled="isPolling || appStore.state.selectedItems.length === 0"
    >
      {{ isPolling ? 'Processing...' : 'Generate Recommendations' }}
    </button>
    <p>Status: {{ status }}</p>
    <table v-if="recommendations && recommendations.length > 0">
      <thead>
        <tr>
          <th>ID</th>
          <th>Recommendation</th>
          <th>Score</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="rec in recommendations" :key="rec.id">
          <td>{{ rec.id }}</td>
          <td>{{ rec.name }}</td>
          <td>{{ rec.score }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup>
import { ref, inject } from 'vue'

const appStore = inject('appStore')

const status = ref('')
const recommendations = ref(null)
const computationId = ref(null)
const isPolling = ref(false)
let pollInterval = null

async function generateRecommendations() {
  try {
    isPolling.value = true
    status.value = ''
    const response = await fetch('/api/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(
        appStore.state.selectedItems.map(({ id, rating, status }) => ({
          id,
          rating,
          status
        }))
      )
    })

    if (!response.ok) throw new Error('Failed to start computation.')

    const data = await response.json()
    computationId.value = data.id
    pollStatus(data.id)
  } catch (error) {
    console.error(error)
    status.value = 'Failed to start computation.'
    isPolling.value = false
  }
}

function pollStatus(id) {
  pollInterval = setInterval(async () => {
    try {
      const response = await fetch(`/api/pollstatus/${id}`)
      if (!response.ok) throw new Error('Failed to poll status.')

      const data = await response.json()
      status.value = data.status

      if (data.status === 'finished') {
        clearInterval(pollInterval)
        pollInterval = null
        fetchResults(id)
      }
    } catch (error) {
      console.error(error)
      status.value = 'Error polling status.'
      clearInterval(pollInterval)
      pollInterval = null
      isPolling.value = false
    }
  }, 3000)
}

async function fetchResults(id) {
  try {
    status.value = 'Fetching results...'
    const response = await fetch(`/api/fetchresults/${id}`)
    if (!response.ok) throw new Error('Failed to fetch results.')

    const results = await response.json()
    if (!results || !Array.isArray(results)) {
      throw new Error('Invalid results from server.')
    }
    recommendations.value = results
    status.value = 'Recommendations ready.'
  } catch (error) {
    console.error(error)
    status.value = 'Failed to fetch results.'
  } finally {
    isPolling.value = false
  }
}
</script>

<style scoped>
@import '../styles/UserRecommendations.css';
</style>
