package com.example.ajay.toolclassification;

import android.annotation.SuppressLint;
import android.annotation.TargetApi;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Build;
import android.provider.MediaStore;
import android.support.annotation.IntDef;
import android.support.annotation.NonNull;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toolbar;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.InputStream;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.util.logging.Logger;

public class MainActivity extends Activity {
    //Load the tensorflow inference library
    static {
        System.loadLibrary("tensorflow_inference");
    }
    static final int REQUEST_IMAGE_CAPTURE = 1;

    //PATH TO OUR MODEL FILE AND NAMES OF THE INPUT AND OUTPUT NODES
    private String MODEL_PATH = "file:///android_asset/tool_classification.pb";
    private String INPUT_NAME = "zero_padding2d_1_input";
    private String OUTPUT_NAME = "output_1";
    private TensorFlowInferenceInterface tf;

    //ARRAY TO HOLD THE PREDICTIONS AND FLOAT VALUES TO HOLD THE IMAGE DATA
    float[] PREDICTIONS = new float[1000];
    private float[] floatValues;
    private int[] INPUT_SIZE = {128,128,3};
    public final int REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS = 1;

    //requests for runtime time permissions
    TextView t1;
    TextView resultView;
    Snackbar progressBar;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setActionBar(toolbar);

        //initialize tensorflow with the AssetManager and the Model
        try {
            System.out.print(MODEL_PATH);
            tf = new TensorFlowInferenceInterface(getAssets(), MODEL_PATH);
            System.out.print(tf);
        }
        catch (Exception e){
            tf = new TensorFlowInferenceInterface(getAssets(), "assets/tool_classification.pb");
            e.printStackTrace();
        }
        t1 =  findViewById(R.id.t1);
        resultView =  findViewById(R.id.results);
        progressBar = Snackbar.make(t1,"PROCESSING IMAGE",Snackbar.LENGTH_INDEFINITE);
        t1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                dispatchTakePictureIntent();
            }
        });

    }

    private void dispatchTakePictureIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            Bundle extras = data.getExtras();
            Bitmap imageBitmap = null;
            if (extras != null) {
                imageBitmap = (Bitmap) extras.get("data");
                progressBar.show();
                predict(imageBitmap);
            }

        }
    }

    //FUNCTION TO COMPUTE THE MAXIMUM PREDICTION AND ITS CONFIDENCE
    public Object[] argmax(float[] array){
        int best = -1;
        float best_confidence = 0.0f;
        for(int i = 0;i < array.length;i++){

            float value = array[i];

            if (value > best_confidence){

                best_confidence = value;
                best = i;
            }
        }

        return new Object[]{best,best_confidence};

    }


    @SuppressLint("StaticFieldLeak")
    public void predict(final Bitmap bitmap) {

        //Runs inference in background thread
        new AsyncTask<Integer, Integer, Integer>() {
            @Override

            protected Integer doInBackground(Integer... params) {
                //Resize the image into 128 x 128
                Bitmap resized_image = ImageUtils.processBitmap(bitmap, 128);
                //Normalize the pixels
                floatValues = ImageUtils.normalizeBitmap(resized_image, 128, 64.5f, 1.0f);

                //Pass input into the tensorflow

                tf.feed(INPUT_NAME, floatValues, 1, 128, 128, 3);
                //compute predictions

                tf.run(new String[]{OUTPUT_NAME});

                //copy the output into the PREDICTIONS array
                tf.fetch(OUTPUT_NAME, PREDICTIONS);

                //Obtained highest prediction
                Object[] results = argmax(PREDICTIONS);
                int class_index = (Integer) results[0];
                float confidence = (Float) results[1];
                String label = "Detected Tool";
                //Convert predicted class index into actual label name

                try {
                    final String conf = String.valueOf(confidence * 100).substring(0, 5);
                    if (class_index == 0 )
                        label = "Scratch Mark";
                    else if(class_index== 1)
                        label = "Slot Damage";
                    else if(class_index== 2)
                        label = "Thinning";
                    else if(class_index== 3)
                        label = "Wrinkle";
                    else if(class_index== 4)
                        label = "0k Back";
                    else if(class_index== 5)
                        label = "ok Front";
                    else
                        label = "Not Able to detect Capture Image, capture photo in Vertical Position";
                    //Display result on UI
                    final String finalLabel = label;
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            progressBar.dismiss();
                            resultView.setText(String.format("%s : %s", finalLabel, conf));
                        }
                    });
                } catch (Exception e) {
                    e.printStackTrace();
                }
                return 0;
            }
        }.execute(0);
    }
}
