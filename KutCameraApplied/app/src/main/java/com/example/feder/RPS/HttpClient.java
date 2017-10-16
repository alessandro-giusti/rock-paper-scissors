package com.example.feder.RPS;

import android.util.Log;

import java.io.File;

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
import retrofit2.http.Header;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.PUT;
import retrofit2.http.Part;
import retrofit2.http.Path;

/**
 * Created by feder on 27.09.2017.
 */


class ServiceGenerator {

    //private static final String BASE_URL = "http://[2001:0:9d38:6abd:15:1230:f5f4:efe9]:8080";
    //private static final String BASE_URL = "http://10.11.16.22:8080";
    private static final String BASE_URL = "http://10.11.16.22:8080";

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
    Call<ResponseBody> updateUser(@Part("label") RequestBody label, @Part MultipartBody.Part image);
}




public class HttpClient {

    public static void uploadImage(File image, String targhetta) {
        UploadService service = ServiceGenerator.createService(UploadService.class);

        RequestBody requestFile = RequestBody.create(MediaType.parse("multipart/form-data"), image);
        MultipartBody.Part body = MultipartBody.Part.createFormData("image", image.getName(), requestFile);

        String targhetta2 = targhetta;
        if (targhetta2 == null) targhetta2 = "error";
        RequestBody label = RequestBody.create(MediaType.parse("multipart/form-data"), targhetta2);
        Call<ResponseBody> call = service.updateUser(label, body);

        call.enqueue(new Callback<ResponseBody>() {
            @Override
            public void onResponse(Call<ResponseBody> call, Response<ResponseBody> response) {
                Log.v("Upload", "success");
            }

            @Override
            public void onFailure(Call<ResponseBody> call, Throwable t) {
                Log.e("Upload error:", t.getMessage());
            }
        });
    }


}
