package com.example.feder.RPS;

import android.util.Log;

import com.google.gson.JsonObject;

import java.io.File;
import java.io.IOException;
import java.util.function.BiFunction;
import java.util.function.Function;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import okhttp3.logging.HttpLoggingInterceptor;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import retrofit2.http.Body;
import retrofit2.http.DELETE;
import retrofit2.http.GET;
import retrofit2.http.Header;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.PUT;
import retrofit2.http.Part;
import retrofit2.http.Path;

/**
 * Created by feder on 27.09.2017.
 */

class lastImage{
    static private String lastImage = "";
    static public void set(String last){
        lastImage = last;


    }
    static String get(){
        return lastImage;
    }
}

class ServiceGenerator {

    //private static final String BASE_URL = "http://[2001:0:9d38:6abd:15:1230:f5f4:efe9]:8080";
    private static final String BASE_URL = "https://rps.idsia.ch";
    //private static final String BASE_URL = "http://192.168.1.107:8000";
    //private static final String BASE_URL = "http://10.11.16.43:8000";

    private static Retrofit.Builder builder =
            new Retrofit.Builder()
                    .baseUrl(BASE_URL)
                    .addConverterFactory(GsonConverterFactory.create());

    private static Retrofit retrofit = builder.build();

    private static HttpLoggingInterceptor logging =new HttpLoggingInterceptor().setLevel(HttpLoggingInterceptor.Level.BODY);

    private static OkHttpClient.Builder httpClient =
            new OkHttpClient.Builder();

    public static <S> S createService(Class<S> serviceClass) {
        if (!httpClient.interceptors().contains(logging)) {
            httpClient.addInterceptor(logging);
            builder.client(httpClient.build());
            retrofit = builder.build();
        }

        return retrofit.create(serviceClass);
    }
}

interface UploadService {

    @Multipart
    @POST("/photo")//THIS IS SO WRONG :/ TODO
    Call<ResponseBody> uploadPhoto(
            @Part("label") RequestBody label,
            @Part("deviceId") RequestBody deviceId,
            @Part MultipartBody.Part image
    );
}
interface TestService {
    @GET("/testString")//THIS IS SO WRONG :/ TODO
    Call<ResponseBody> testString();
    @GET("/test")//THIS IS SO WRONG :/ TODO
    Call<ResponseBody> test();
}


interface DeleteService {
    @DELETE("/photo/{id}")//THIS IS SO WRONG :/ TODO
    Call<ResponseBody> delete(@Path("id") String id);
}




public class HttpClient {

    private final static String TAG = "RPS";

    public static void uploadImage(File image, String targhetta, String devId, Command toDoAfter,Command ifFailure) {
        UploadService service = ServiceGenerator.createService(UploadService.class);

        RequestBody requestFile = RequestBody.create(MediaType.parse("multipart/form-data"), image);
        MultipartBody.Part body = MultipartBody.Part.createFormData("image", image.getName(), requestFile);

        String targhetta2 = targhetta;
        if (targhetta2 == null) targhetta2 = "error";
        RequestBody label = RequestBody.create(MediaType.parse("multipart/form-data"), targhetta2);


        String devId2 = devId;
        if (devId2 == null) devId2 = "error";
        RequestBody deviceId = RequestBody.create(MediaType.parse("multipart/form-data"), devId2);

        Call<ResponseBody> call = service.uploadPhoto(label, deviceId,body);



        call.enqueue(new Callback<ResponseBody>() {
            @Override
            public void onResponse(Call<ResponseBody> call, Response<ResponseBody> response) {
                ResponseBody body =response.body();
                try {
                    String s = body.string();
                    //s = s.replace("\"","");
                    lastImage.set(s);
                    Log.v(TAG, "Upload success: "+s);
                    toDoAfter.execute();

                } catch (IOException|NullPointerException e) {
                    Log.e(TAG, "Failure");
                    ifFailure.execute();
                }

                //JsonObject post = new JsonObject().get(body.toString()).getAsJsonObject();
                //Log.d(TAG, post.toString());


            }

            @Override
            public void onFailure(Call<ResponseBody> call, Throwable t) {
                ifFailure.execute();
                Log.e(TAG, "Upload error:"+ t.getMessage());
            }
        });

    }

    public static void testImage(File image, String devId, Command finalfunc,Command ifFailure) {
        UploadService service = ServiceGenerator.createService(UploadService.class);

        RequestBody requestFile = RequestBody.create(MediaType.parse("multipart/form-data"), image);
        MultipartBody.Part body = MultipartBody.Part.createFormData("image", image.getName(), requestFile);

        String targhetta2 = "testing";
        if (targhetta2 == null) targhetta2 = "error";
        RequestBody label = RequestBody.create(MediaType.parse("multipart/form-data"), targhetta2);


        String devId2 = devId;
        if (devId2 == null) devId2 = "error";
        RequestBody deviceId = RequestBody.create(MediaType.parse("multipart/form-data"), devId2);

        Call<ResponseBody> call = service.uploadPhoto(label, deviceId,body);



        call.enqueue(new Callback<ResponseBody>() {
            @Override
            public void onResponse(Call<ResponseBody> call, Response<ResponseBody> response) {
                ResponseBody body =response.body();
                try {
                    String s = body.string();
                    s = s.replace("[","");
                    s = s.replace("]","");
                    String[] probs = s.split(",");
                    float pr = Float.valueOf(probs[0]);
                    float pp = Float.valueOf(probs[1]);
                    float ps = Float.valueOf(probs[2]);
                    finalfunc.execute((int) (pp * 100),(int) (pr * 100),(int) (ps * 100));

                    Log.v(TAG, "Upload success: " + s);

                } catch (IOException|NullPointerException e) {
                    Log.e(TAG, "Failure converting body to string");
                }

                //JsonObject post = new JsonObject().get(body.toString()).getAsJsonObject();
                //Log.d(TAG, post.toString());



            }

            @Override
            public void onFailure(Call<ResponseBody> call, Throwable t) {
                ifFailure.execute();
                Log.e(TAG, "Upload error:"+ t.getMessage());
            }
        });

    }
    public static void testConnection(Command ifSuccess, Command ifFailure) {
        TestService service = ServiceGenerator.createService(TestService.class);

        Call<ResponseBody> call1 = service.test();

        call1.enqueue(new Callback<ResponseBody>() {
            @Override
            public void onResponse(Call<ResponseBody> call, Response<ResponseBody> response) {
                try{
                    ResponseBody body =response.body();
                    String s = body.string();
                    Log.v(TAG, "Test success: " + s);
                    ifSuccess.execute();
                } catch (IOException|NullPointerException e) {
                    Log.e(TAG, "Failure converting body to string");
                    ifFailure.execute();
                }


        }

            @Override
            public void onFailure(Call<ResponseBody> call, Throwable t) {
                Log.e(TAG, "Upload error:"+ t.getMessage());
                ifFailure.execute();
            }
        });


    }


    public static void deletePhoto(Command ifSuccess, Command ifFailure) {
        DeleteService service = ServiceGenerator.createService(DeleteService.class);

        Call<ResponseBody> call = service.delete(lastImage.get());

        call.enqueue(new Callback<ResponseBody>() {
            @Override
            public void onResponse(Call<ResponseBody> call, Response<ResponseBody> response) {
                ResponseBody body =response.body();
                try {
                    Log.d(TAG, body.string());
                    ifSuccess.execute();
                } catch (IOException|NullPointerException e) {
                    ifFailure.execute();
                    Log.e(TAG, "Failure converting body to string");
                }

                //JsonObject post = new JsonObject().get(body.toString()).getAsJsonObject();
                //Log.d(TAG, post.toString());



            }

            @Override
            public void onFailure(Call<ResponseBody> call, Throwable t) {
                Log.e(TAG, "Upload error:"+ t.getMessage());
                ifFailure.execute();
            }
        });

    }

}
