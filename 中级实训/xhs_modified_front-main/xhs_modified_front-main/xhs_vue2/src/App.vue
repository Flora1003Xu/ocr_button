<template>
  <div id="app">
    <div>
    <router-view></router-view>
    </div>
  </div>
</template>

<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 0px;
}
.el-tabs__item {
  color: #000 !important;
}
.el-tabs__item.is-active {
  color: black !important;
  font-weight:600;
}
.el-table tbody tr:hover>td {
  background-color:rgb(255,255,255) !important
}
</style>

<script>
export default{
  name:'app',
}
</script>