import mongoose from 'mongoose';

export const connectDB = async () => {
  const dbUrl = process.env.MONGO_URI;
  try {
    if (!dbUrl) {
      throw new Error('db connection url is not found!');
    }

    const conn = await mongoose.connect(dbUrl.toString());
    console.log(`db connected successfully`);
    return conn;
  } catch (error) {
    console.error(error?.message);
    process.exit(1);
  }
};
