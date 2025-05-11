import mongoose from "mongoose";
  
const authSchema = new mongoose.Schema({
    username: {
        type: String,
        required: true,
        unique: true,  
        trim: true,    
    },
    password: {
        type: String,
        required: true,
        minlength: 6,  
    },
});
const messageSchema = new mongoose.Schema({
    username:{type: String},
    query: { type: String },
    imageBase64: { type: String },
    res: { type: String },
    desc:{type:String},
    doctors: { type: Array, default: [] }
});
export const chatAuthData = mongoose.model("chatAuthData" , authSchema);

export const myMessages = mongoose.model("myMessages" , messageSchema) ; 