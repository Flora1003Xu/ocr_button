import Vue from 'vue'
import VueRouter from 'vue-router'
import PersonalView from '../views/PersonalView.vue'
//import { component } from 'vue/types/umd'

Vue.use(VueRouter)

const originalPush = VueRouter.prototype.push
VueRouter.prototype.push = function push(location) {
  return originalPush.call(this, location).catch(err => err)
}

export const routes = [
  {
    path:'/',
    name:'personal',
    component:PersonalView,
    meta:{title:'personal',roles:['user']}
  }
]

const router = new VueRouter({
  routes
})

export default router
