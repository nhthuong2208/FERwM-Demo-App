import axios from "axios"

const baseRequest = axios.create({
    baseURL: "http://127.0.0.1:5000/",
    headers: ""
})

export default baseRequest;