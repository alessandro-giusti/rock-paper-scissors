package com.example.feder.RPS;


import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Calendar;


import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.hardware.Camera;
import android.hardware.Camera.CameraInfo;
import android.hardware.Camera.ErrorCallback;
import android.hardware.Camera.PictureCallback;
import android.hardware.Camera.ShutterCallback;
import android.media.ExifInterface;
import android.os.Build;
import android.os.Bundle;
import android.app.Activity;
import android.provider.Settings;
import android.support.v4.app.FragmentActivity;
import android.support.v7.app.AppCompatActivity;
import android.text.Layout;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.Spinner;
import android.widget.SpinnerAdapter;
import android.widget.TextView;

public class MainActivity extends Activity {
    private final static String TAG = "RPS";
    ImageView imageSmallPreview;
    ImageView labelGiver;
    Bitmap image;
	String label;
    SurfaceView previewSurface;
	Activity context;
	Preview preview;
	Camera camera;
    ImageView fotoButton;
    Spinner sp;
    ArrayAdapter<String> spAdp;
    ImageView[] tests;
	LinearLayout progressLayout;
    ImageView leftButton;
    ImageView rightButton;
    ImageView deleteButton;
	String path = "/sdcard/KutCamera/cache/images/";

	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		
		context=this;
        image = null;
        label = "Rock";

		fotoButton = (ImageView) findViewById(R.id.take_pic);
        sp = (Spinner) findViewById(R.id.spinner);
        spAdp = new ArrayAdapter<String>(this, R.layout.row);
        labelGiver =  (ImageView) findViewById(R.id.label_view);
        imageSmallPreview = (ImageView) findViewById(R.id.image_small_preview);
		progressLayout = (LinearLayout) findViewById(R.id.progress_layout);
        previewSurface=(SurfaceView) findViewById(R.id.KutCameraFragment);
		preview = new Preview(this,previewSurface);
		FrameLayout frame = (FrameLayout) findViewById(R.id.preview);
		frame.addView(preview);
		preview.setKeepScreenOn(true);

        spAdp.addAll(new String[]{"Training, Testing"});
        sp.setAdapter(spAdp);

        sp.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener()
        {

            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int arg2, long arg3) {
                TextView txt=(TextView) arg1.findViewById(R.id.rowtext);
                String s=txt.getText().toString();
                if (s.equals("Training")){}
                else{}//TODO


            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            { }
        });

        tests=new ImageView[]{
                (ImageView) findViewById(R.id.image_test1),
                (ImageView) findViewById(R.id.image_test2),
                (ImageView) findViewById(R.id.image_test3),
                (ImageView) findViewById(R.id.image_test4),
                (ImageView) findViewById(R.id.image_test5),
                (ImageView) findViewById(R.id.image_test6),
                (ImageView) findViewById(R.id.image_test7),
                (ImageView) findViewById(R.id.image_test8),
                (ImageView) findViewById(R.id.image_test9)
        };



		fotoButton.setOnClickListener(new OnClickListener() {

			@Override
			public void onClick(View v) {

				try {
					takeFocusedPicture();
				} catch (Exception e) {

				}

				fotoButton.setClickable(false);
				progressLayout.setVisibility(View.VISIBLE);
			}
		});

		imageSmallPreview.setOnClickListener(new OnClickListener() {

			@Override
			public void onClick(View v) {


			}
		});

        imageSmallPreview.setOnLongClickListener(new View.OnLongClickListener() {
            @Override
            public boolean onLongClick(View view) {
                return false;
            }
        });



        labelGiver.setOnClickListener(new OnClickListener() {

            @Override
            public void onClick(View v) {

                if (labelGiver.getTag().toString().equals("R")) {
                    labelGiver.setImageResource(R.drawable.rps_p);
                    labelGiver.setTag("P");
					label = "Paper";
                } else if (labelGiver.getTag().toString().equals("P")){
                    labelGiver.setImageResource(R.drawable.rps_s);
                    labelGiver.setTag("S");
					label = "Scissors";
                }  else if (labelGiver.getTag().toString().equals("S")){
                    labelGiver.setImageResource(R.drawable.rps_r);
                    labelGiver.setTag("R");
					label = "Rock";
                }
            }
        });


	}



	@Override
	protected void onResume() {
		super.onResume();
		// TODO Auto-generated method stub
		if(camera==null){
		camera = Camera.open();
		camera.startPreview();
		camera.setErrorCallback(new ErrorCallback() {
			public void onError(int error, Camera mcamera) {

				camera.release();
				camera = Camera.open();
				Log.e(TAG, "error camera");

			}
		});
		}
		if (camera != null) {
			if (Build.VERSION.SDK_INT >= 14)
				setCameraDisplayOrientation(context,
						CameraInfo.CAMERA_FACING_BACK, camera);
			preview.setCamera(camera);
		}
	}
	
	private void setCameraDisplayOrientation(Activity activity, int cameraId,
			android.hardware.Camera camera) {
		android.hardware.Camera.CameraInfo info = new android.hardware.Camera.CameraInfo();
		android.hardware.Camera.getCameraInfo(cameraId, info);
		int rotation = activity.getWindowManager().getDefaultDisplay()
				.getRotation();
		int degrees = 0;
		switch (rotation) {
		case Surface.ROTATION_0:
			degrees = 0;
			break;
		case Surface.ROTATION_90:
			degrees = 90;
			break;
		case Surface.ROTATION_180:
			degrees = 180;
			break;
		case Surface.ROTATION_270:
			degrees = 270;
			break;
		}

		int result;
		if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
			result = (info.orientation + degrees) % 360;
			result = (360 - result) % 360; // compensate the mirror
		} else { // back-facing
			result = (info.orientation - degrees + 360) % 360;
		}
		camera.setDisplayOrientation(result);
	}


	
	Camera.AutoFocusCallback mAutoFocusCallback = new Camera.AutoFocusCallback() {
		@Override
		public void onAutoFocus(boolean success, Camera camera) {
				
			try{
			camera.takePicture(mShutterCallback, null, jpegCallback);
			}catch(Exception e){
				
			}

		}
	};

	Camera.ShutterCallback mShutterCallback = new ShutterCallback() {
		
		@Override
		public void onShutter() {
			// TODO Auto-generated method stub
			
		}
	};
	public void takeFocusedPicture() {
		camera.autoFocus(mAutoFocusCallback);

	}

	PictureCallback rawCallback = new PictureCallback() {
		public void onPictureTaken(byte[] data, Camera camera) {
			// Log.d(TAG, "onPictureTaken - raw");
		}
	};

	PictureCallback jpegCallback = new PictureCallback() {
		@SuppressWarnings("deprecation")
		public void onPictureTaken(byte[] data, Camera camera) {


			Calendar c = Calendar.getInstance();
			File fotoDir = new File(path);

            File foto = new File(path + label + "_" + c.getTime().getDate() + c.getTime().getHours() + c.getTime().getMinutes() + c.getTime().getSeconds() + ".jpg");
			ByteArrayOutputStream stream = new ByteArrayOutputStream();


            if (!foto.exists()) {
                try {
                    foto.createNewFile();
                } catch (IOException e) {
                    Log.e(TAG, "Error creating file: " + e.getMessage() );


                }
            }

			if (!fotoDir.exists()) {
                fotoDir.mkdirs();
			}

			try(FileOutputStream outStream= new FileOutputStream(foto)) {
				//write bitmap
                image = BitmapFactory.decodeByteArray(data, 0, data.length);


                // to crop bmp to a square (image, Xstart, Ystart, width, height)
                int minSize = image.getWidth() < image.getHeight()? image.getWidth() : image.getHeight();
                image=Bitmap.createBitmap(image, image.getWidth()/2 - minSize / 2,image.getHeight()/2 - minSize / 2,minSize, minSize);


                //resize
                image= Bitmap.createScaledBitmap(image, 500, 500, true);


                // Convert image to JPEG

                image.compress(Bitmap.CompressFormat.JPEG, 100, stream);
                byte[] byteArray = stream.toByteArray();
                image = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);

                // Write to SD Card
				outStream.write(byteArray);
				outStream.close();





			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			} finally {

			}
			
			
			Bitmap imagePreview;
			 final BitmapFactory.Options options = new BitmapFactory.Options();
			  options.inSampleSize = 5;
			   
			    options.inPurgeable=true;                   //Tell to gc that whether it needs free memory, the Bitmap can be cleared

			    options.inInputShareable=true;              //Which kind of reference will be used to recover the Bitmap data after being clear, when it will be used in the future


			byte[] byteArray = stream.toByteArray();
			imagePreview = BitmapFactory.decodeByteArray(byteArray,0,byteArray.length,options);



			imageSmallPreview.setImageBitmap(imagePreview);
            String devId=Settings.Secure.getString(getApplicationContext().getContentResolver(), Settings.Secure.ANDROID_ID);

            try {
                HttpClient.uploadImage(foto,label, devId);
            } catch (Exception e) {
                Log.e(TAG, "Error uploading: " + e.getMessage() );

            }


            if ( label.equals("Paper")) {
                for (ImageView test : tests){
                    test.setImageResource(R.drawable.rps_p);
                }
            } else if ( label.equals("Rock")) {
                for (ImageView test : tests){
                    test.setImageResource(R.drawable.rps_r);
                }
            }  else if ( label.equals("Scissors")) {
                for (ImageView test : tests){
                    test.setImageResource(R.drawable.rps_s);
                }
            }

			fotoButton.setClickable(true);
			camera.startPreview();
			progressLayout.setVisibility(View.GONE);


		}
	};

	public static Bitmap rotate(Bitmap source, float angle) {
		Matrix matrix = new Matrix();
		matrix.postRotate(angle);
		return Bitmap.createBitmap(source, 0, 0, source.getWidth(),
				source.getHeight(), matrix, false);
	}

}
