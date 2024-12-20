// router/index.js
import { createRouter, createWebHistory } from 'vue-router'
import GameLibrary from '../components/GameLibrary.vue'
import UserRecommendations from '../components/UserRecommendations.vue'

const routes = [
  { path: '/', redirect: '/library' },
  { path: '/library', component: GameLibrary },
  { path: '/recommendations', component: UserRecommendations }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
