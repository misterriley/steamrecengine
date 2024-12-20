<!-- components/AppHeader.vue -->
<template>
  <header>
    <h1>Recommendation System</h1>
    <nav>
      <router-link to="/library">GameLibrary</router-link>
      <router-link to="/recommendations">UserRecommendations</router-link>
    </nav>
  </header>
</template>

<script setup>
</script>

<style scoped>
@import '../styles/AppHeader.css';
</style>
