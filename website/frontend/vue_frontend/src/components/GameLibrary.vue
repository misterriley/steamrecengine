<!-- components/GameLibrary.vue -->
<template>
  <div>
    <h2>Library</h2>
    <div class="search-bar-container">
      <input
        type="text"
        v-model="searchText"
        @input="handleSearch"
        placeholder="Search for an item..."
        class="search-input"
      />
      <ul v-if="filteredItems.length > 0" class="dropdown">
        <li v-for="item in filteredItems" :key="item.id" @click="handleItemClick(item.name)">
          {{ item.name }}
        </li>
      </ul>
    </div>

    <table v-if="appStore.state.selectedItems.length > 0" class="item-table">
      <thead>
        <tr>
          <th>Name</th>
          <th>Game ID</th>
          <th>Rating</th>
          <th>More like this</th>
          <th>Less like this</th>
          <th>Ignore</th>
          <th>Delete</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(item, index) in appStore.state.selectedItems" :key="item.id">
          <td>{{ item.name }}</td>
          <td>{{ item.id }}</td>
          <td>
            <input
              type="number"
              v-model="item.rating"
              placeholder="Rating"
              @input="handleInputChange(index, 'rating', item.rating)"
            />
          </td>
          <td>
            <input
              type="radio"
              :checked="item.status === 'more'"
              @change="handlePreferenceChange(index, 'more')"
            />
          </td>
          <td>
            <input
              type="radio"
              :checked="item.status === 'less'"
              @change="handlePreferenceChange(index, 'less')"
            />
          </td>
          <td>
            <input
              type="radio"
              :checked="item.status === 'ignore'"
              @change="handlePreferenceChange(index, 'ignore')"
            />
          </td>
          <td>
            <button @click="handleDelete(index)">Delete</button>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useAppStore } from '../store/appStore';

const appStore = useAppStore();

const searchText = ref('')
const filteredItems = ref([])
const items = ref([])

onMounted(() => {
  fetch('https://raw.githubusercontent.com/misterriley/steamrecengine/refs/heads/main/games.json')
    .then(response => response.json())
    .then(data => {
      items.value = data
    })
    .catch(error => console.error('Error fetching data:', error))
})

function handleSearch() {
  const text = searchText.value
  if (text.length >= 2) {
    const matches = items.value.filter(item =>
      item.name.toLowerCase().includes(text.toLowerCase())
    )
    filteredItems.value = matches.map(item => ({ name: item.name, id: item.game_id }))
  } else {
    filteredItems.value = []
  }
}

function handleItemClick(itemName) {
  const selectedItem = filteredItems.value.find(item => item.name === itemName)
  if (selectedItem) {
    appStore.addItem({
      name: selectedItem.name,
      id: selectedItem.id,
      rating: '',
      status: 'more'
    })
  }
  searchText.value = ''
  filteredItems.value = []
}

function handleInputChange(index, field, value) {
  appStore.updateItem(index, { [field]: value })
}

function handlePreferenceChange(index, status) {
  appStore.updateItem(index, { status })
}

function handleDelete(index) {
  appStore.deleteItem(index)
}
</script>

<style scoped>
@import '../styles/GameLibrary.css';
</style>
