package ch.supsi;


import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.servlet.http.HttpServletResponse;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;

@RestController
public class PhotoController {



	@CrossOrigin
	@RequestMapping(value = "/photo", method = RequestMethod.POST)
	public String postPhoto(@RequestParam("file") MultipartFile file, HttpServletResponse response) {




			String name = "TEST";
			if (!file.isEmpty()) {
				try {
					byte[] bytes = file.getBytes();
					BufferedOutputStream stream = new BufferedOutputStream(new FileOutputStream(new File(name)));
					stream.write(bytes);
					stream.close();
					return "You successfully uploaded " + name;
				} catch (Exception e) {
					return "You failed to upload " + name + " => " + e.getMessage();
				}
			} else {
				return "You failed to upload " + name + " because the file was empty.";
			}

	}

	@CrossOrigin
	@RequestMapping(value = "/photos", method = RequestMethod.GET)
	public void getPhotos(HttpServletResponse response) {

	}

	@CrossOrigin
	@RequestMapping(value = "/photo/{photoName}", method = RequestMethod.GET)
	public void getPhoto(@PathVariable String photoName, HttpServletResponse response) {

	}

	@CrossOrigin
	@RequestMapping(value = "/photo/{photoName}", method = RequestMethod.DELETE)
	public void deletePhoto(@PathVariable String photoName, HttpServletResponse response) {

	}

}
