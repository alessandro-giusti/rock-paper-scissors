package ch.idsia.robotics.RPS;


import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Calendar;


import android.app.AlertDialog;
import android.content.DialogInterface;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.hardware.Camera;
import android.hardware.Camera.CameraInfo;
import android.hardware.Camera.ErrorCallback;
import android.hardware.Camera.PictureCallback;
import android.hardware.Camera.ShutterCallback;
import android.os.Build;
import android.os.Bundle;
import android.app.Activity;
import android.provider.Settings;
import android.support.graphics.drawable.VectorDrawableCompat;
import android.support.v7.app.AppCompatDelegate;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.CompoundButton;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ToggleButton;


public class MainActivity extends Activity {
    static {
        AppCompatDelegate.setCompatVectorFromResourcesEnabled(true);
    }
    private final static String TAG = "RPS";
    private final static int imgSize = 500;
    ImageView imageSmallPreview;

    String currentPhotoPath;
    ImageView labelGiver;
    ToggleButton toggle;
    Bitmap image;
    Bitmap imagePreview;
    String label;
    //TODO FOR DIFFERENT PREVIEW FrameLayout cameraContainer;
    SurfaceView previewSurface;
    Activity context;
    Preview preview;
    //TODO FOR DIFFERENT PREVIEW Preview2 preview;
    Camera camera;
    ImageView fotoButton;
    ImageView bigPreview;
    ImageView connectionStatus;
    ImageView givenLabel;
    LinearLayout progressLayout;
    String path = "/sdcard/KutCamera/cache/images/";
    Boolean isTraining;
    ImageView deleteButton;
    View[] disabledDuringLoading;



    class deleteCommand extends Command
    {
        public void execute()
        {
            imageSmallPreview.setImageDrawable(null);
            givenLabel.setImageDrawable(null);
            File file = new File(currentPhotoPath);
            boolean deleted = file.delete();
            deleteButton.setVisibility(View.INVISIBLE);


        }


    }

    class testUpdates extends Command
    {
        public void execute( int pp, int pr, int ps)
        {

            image = overlay(image, generateGraph(pr,pp,ps));
            bigPreview.setImageBitmap(image);
            new Connected().execute();

        }


    }


    class trainUpdates extends Command
    {
        final private String labelAtClick;

        trainUpdates(String label){
            labelAtClick=label;
        }
        @Override
        public void execute(){

            imageSmallPreview.setImageBitmap(imagePreview);

            if (labelAtClick.equals("Paper")) {
                givenLabel.setImageResource(R.drawable.rps_p);
            } else if (labelAtClick.equals("Rock")) {
                givenLabel.setImageResource(R.drawable.rps_r);
            } else if (labelAtClick.equals("Scissors")) {
                givenLabel.setImageResource(R.drawable.rps_s);
            }
            deleteButton.setVisibility(View.VISIBLE);

            new Connected().execute();

        }


    }






    class NotConnected extends Command
    {



        public void execute()
        {


            AlertDialog mydialog = generateDialog("Errore di connessione","Assicurati di essere connesso ad internet");
            mydialog.show();
            connectionStatus.setImageResource(R.drawable.no_conn);
            enableThings();


        }
    }

    class Connected extends Command
    {



        public void execute()
        {


            connectionStatus.setImageResource(R.drawable.yes_conn);
            enableThings();

        }
    }



    public void enableThings()
    {


        for (View x : disabledDuringLoading) x.setClickable(true);
        camera.startPreview();
        progressLayout.setVisibility(View.GONE);

    }
    public AlertDialog generateDialog(String text, String title){
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setMessage(text).setTitle(title);
        builder.setPositiveButton("Ok", new AlertDialog.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialogInterface, int i) {

            }
        });

        AlertDialog dialog = builder.create();
        return dialog;
    }

    public void checkConnection(){
	    HttpClient.testConnection(new Connected(), new NotConnected());

    }

    private Bitmap overlay(Bitmap bmp1, Bitmap bmp2) {
        Bitmap bmOverlay = Bitmap.createBitmap(bmp1.getWidth(), bmp1.getHeight(), bmp1.getConfig());
        Canvas canvas = new Canvas(bmOverlay);
        canvas.drawBitmap(bmp1, new Matrix(), null);
        canvas.drawBitmap(bmp2, 0,0, null);
        return bmOverlay;
    }
    private Bitmap generateGraph(int pr,int pp,int ps) {

        Rect background = new Rect(0, 0, imgSize, imgSize);

        Bitmap image = Bitmap.createBitmap(background.width(), background.height(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(image);

        Paint paint = new Paint();
        paint.setColor(Color.TRANSPARENT);

        Paint bianco = new Paint();
        bianco.setColor(Color.WHITE);

        Paint nero = new Paint();
        nero.setColor(Color.BLACK);

        int rectWidth=imgSize/13;
        int rectWidth2=imgSize/14;

        int paperHeight = pp * (imgSize/100);
        int rockHeight = pr * (imgSize/100);
        int scissorHeight = ps * (imgSize/100);

        Rect rock = new Rect(imgSize*2/6 - rectWidth, (imgSize-rockHeight/2) - imgSize/7, imgSize*2/6 + rectWidth, imgSize-imgSize/7 );
        Rect paper = new Rect(imgSize*3/6 - rectWidth, (imgSize-paperHeight/2) - imgSize/7, imgSize*3/6 + rectWidth, imgSize- imgSize/7);
        Rect scissor = new Rect(imgSize*4/6 - rectWidth, (imgSize-scissorHeight/2) - imgSize/7, imgSize*4/6 + rectWidth, imgSize- imgSize/7);

        int rockBorder = 0;
        if (rockHeight> 30 ) rockBorder = 10;
        int paperBorder = 0;
        if (paperHeight> 30 ) paperBorder = 10;
        int scissorBorder = 0;
        if (scissorHeight> 30 ) scissorBorder = 10;


        Rect rock2 = new Rect(imgSize*2/6 - rectWidth2, (imgSize-rockHeight/2)- imgSize/7 + rockBorder, imgSize*2/6 + rectWidth2, imgSize- imgSize/7 - rockBorder);
        Rect paper2 = new Rect(imgSize*3/6 - rectWidth2, (imgSize-paperHeight/2)- imgSize/7 + paperBorder, imgSize*3/6 + rectWidth2, imgSize- imgSize/7 - paperBorder);
        Rect scissor2 = new Rect(imgSize*4/6 - rectWidth2, (imgSize-scissorHeight/2)- imgSize/7 + scissorBorder, imgSize*4/6 + rectWidth2, imgSize- imgSize/7 -scissorBorder );


        canvas.drawRect(background, paint);
        canvas.drawRect(rock, bianco);
        canvas.drawRect(paper, bianco);
        canvas.drawRect(scissor, bianco);
        canvas.drawRect(rock2, nero);
        canvas.drawRect(paper2, nero);
        canvas.drawRect(scissor2, nero);

        VectorDrawableCompat vR = VectorDrawableCompat.create(getResources(), R.drawable.rps_r, null );
        vR.setBounds(0,0, rectWidth*2,rectWidth*2);
        canvas.translate(imgSize*2/6 - rectWidth, (imgSize-imgSize/7 )  );
        vR.draw(canvas);
        canvas.translate(-(imgSize*2/6 - rectWidth), -(imgSize-imgSize/7 )  );
        VectorDrawableCompat vP = VectorDrawableCompat.create(getResources(), R.drawable.rps_p, null );
        vP.setBounds(0,0, rectWidth*2,rectWidth*2);
        canvas.translate(imgSize*3/6 - rectWidth, (imgSize-imgSize/7 )  );
        vP.draw(canvas);
        canvas.translate(-(imgSize*3/6 - rectWidth), -(imgSize-imgSize/7 )  );
        VectorDrawableCompat vS = VectorDrawableCompat.create(getResources(), R.drawable.rps_s, null );
        vS.setBounds(0,0, rectWidth*2,rectWidth*2);
        canvas.translate(imgSize*4/6 - rectWidth, (imgSize-imgSize/7 )  );
        vS.draw(canvas);
        canvas.translate(-(imgSize*4/6 - rectWidth), -(imgSize-imgSize/7 )  );
        //Bitmap bitR = BitmapFactory.decodeResource(getResources(), R.drawable.rps_r);
        //Bitmap bitP = BitmapFactory.decodeResource(getResources(), R.drawable.rps_p);
        //Bitmap bitS = BitmapFactory.decodeResource(getResources(), R.drawable.rps_s);
        //canvas.drawBitmap(bitR,imgSize*2/6 - rectWidth,imgSize-imgSize/10 ,null );
        //canvas.drawBitmap(bitP,imgSize*3/6 - rectWidth,imgSize-imgSize/10 ,null );
       // canvas.drawBitmap(bitS,imgSize*4/6 - rectWidth,imgSize-imgSize/10 ,null );

        return image;
    }



	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);



		isTraining = true;
		context=this;
        image = null;
        label = "Rock";
        currentPhotoPath="";
        connectionStatus = (ImageView)findViewById(R.id.connection_status);
        bigPreview = (ImageView)findViewById(R.id.image_big_preview);
        givenLabel = (ImageView) findViewById(R.id.image_small_result);
		fotoButton = (ImageView) findViewById(R.id.take_pic);
        toggle =  (ToggleButton) findViewById(R.id.testortrain);

        labelGiver =  (ImageView) findViewById(R.id.label_view);
        imageSmallPreview = (ImageView) findViewById(R.id.image_small_preview);
        deleteButton = (ImageView) findViewById(R.id.deletebutton);
		progressLayout = (LinearLayout) findViewById(R.id.progress_layout);

		//REMOVE FOR DIFFERENT PREVIEW
		previewSurface=(SurfaceView) findViewById(R.id.KutCameraFragment);
		preview = new Preview(this,previewSurface);
		FrameLayout frame = (FrameLayout) findViewById(R.id.preview);
		frame.addView(preview);
		preview.setKeepScreenOn(true);
        //END REMOVE FOR DIFFERENT PREVIEW

        //TODO FOR DIFFERENT PREVIEW cameraContainer = (FrameLayout) findViewById(R.id.preview);



        deleteButton.setVisibility(View.INVISIBLE);
        disabledDuringLoading=new View[]{deleteButton,toggle,fotoButton,labelGiver};

        toggle.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {

                if (isChecked) {
                    isTraining=false;
                    labelGiver.setImageDrawable(null);
                    imageSmallPreview.setImageDrawable(null);
                    givenLabel.setVisibility(View.INVISIBLE);
                    deleteButton.setVisibility(View.INVISIBLE);

                } else {
                    isTraining=true;
                    labelGiver.performClick();
                    givenLabel.setVisibility(View.VISIBLE);
                    givenLabel.setImageDrawable(null);
                    deleteButton.setVisibility(View.INVISIBLE);
                }
            }
        });






		fotoButton.setOnClickListener(new OnClickListener() {

			@Override
			public void onClick(View v) {

				try {
					takeFocusedPicture();
				} catch (Exception e) {

				}



                bigPreview.setImageDrawable(null);
                for (View x : disabledDuringLoading) x.setClickable(false);
			}
		});


        bigPreview.setOnClickListener(new OnClickListener() {

            @Override
            public void onClick(View v) {


                bigPreview.setImageDrawable(null);
            }
        });

        deleteButton.setOnClickListener(new OnClickListener() {

            @Override
            public void onClick(View v) {

                HttpClient.deletePhoto(new deleteCommand(), new NotConnected());

            }
        });




        labelGiver.setOnClickListener(new OnClickListener() {

            @Override
            public void onClick(View v) {

                if (isTraining){


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
                else{
                }
            }
        });

       

	}



	@Override
	protected void onResume() {
		super.onResume();
        checkConnection();

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
            //REMOVE FOR CHANGE PREVIEW

			preview.setCamera(camera);

            //END REMOVE FOR CHANGE PREVIEW


            //TODO FOR CHANGE PREVIEW preview = new Preview2(this,camera);
            //TODO FOR CHANGE PREVIEW  cameraContainer.addView(preview);
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

            progressLayout.setVisibility(View.VISIBLE);
			Calendar c = Calendar.getInstance();
			File fotoDir = new File(path);

            //File fotoriginal = new File(path + "original_" + label + "_" + c.getTime().getDate() + c.getTime().getHours() + c.getTime().getMinutes() + c.getTime().getSeconds() + ".jpg");
            currentPhotoPath=path + label + "_" + c.getTime().getDate() + c.getTime().getHours() + c.getTime().getMinutes() + c.getTime().getSeconds() + ".jpg";
            File foto = new File(currentPhotoPath);

            ByteArrayOutputStream stream = new ByteArrayOutputStream();


            if (!foto.exists()) {
                try {
                    foto.createNewFile();
                    //fotoriginal.createNewFile();
                } catch (IOException e) {
                    Log.e(TAG, "Error creating file: " + e.getMessage() );


                }
            }

			if (!fotoDir.exists()) {
                fotoDir.mkdirs();
			}

           /* try(FileOutputStream outStream= new FileOutputStream(fotoriginal)) {
                //write bitmap
                image = BitmapFactory.decodeByteArray(data, 0, data.length);



                image.compress(Bitmap.CompressFormat.JPEG, 100, stream);
                byte[] byteArray = stream.toByteArray();
                // Write to SD Card
                outStream.write(byteArray);
                outStream.close();





            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            } finally {

            }*/
            try(FileOutputStream outStream= new FileOutputStream(foto)) {
                //write bitmap


                image = BitmapFactory.decodeByteArray(data, 0, data.length);



                // to crop bmp to a square (image, Xstart, Ystart, width, height)
                int minSize = image.getWidth() < image.getHeight()? image.getWidth() : image.getHeight();
                image=Bitmap.createBitmap(image, image.getWidth()/2 - minSize / 2,image.getHeight()/2 - minSize / 2,minSize, minSize);


                //resize
                image= Bitmap.createScaledBitmap(image, imgSize, imgSize, true);


                Matrix matrix = new Matrix();
                matrix.postRotate(90);
                image = Bitmap.createBitmap(image , 0, 0, image.getWidth(), image.getHeight(), matrix, true);

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



			 final BitmapFactory.Options options = new BitmapFactory.Options();
			  options.inSampleSize = 5;
			   
			    options.inPurgeable=true;                   //Tell to gc that whether it needs free memory, the Bitmap can be cleared

			    options.inInputShareable=true;              //Which kind of reference will be used to recover the Bitmap data after being clear, when it will be used in the future


			byte[] byteArray = stream.toByteArray();
			imagePreview = BitmapFactory.decodeByteArray(byteArray,0,byteArray.length,options);







            String devId=Settings.Secure.getString(getApplicationContext().getContentResolver(), Settings.Secure.ANDROID_ID);



            try {
                if (isTraining) {
						HttpClient.uploadImage(foto, label, devId, new trainUpdates(label), new NotConnected());

				}
                else {

                    	HttpClient.testImage(foto, devId, new testUpdates(), new NotConnected());

                }
            } catch (Exception e) {
                Log.e(TAG, "Error uploading: " + e.getMessage() );

            }







		}
	};

	public static Bitmap rotate(Bitmap source, float angle) {
		Matrix matrix = new Matrix();
		matrix.postRotate(angle);
		return Bitmap.createBitmap(source, 0, 0, source.getWidth(),
				source.getHeight(), matrix, false);
	}

}
