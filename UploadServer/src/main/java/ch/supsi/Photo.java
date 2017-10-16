package ch.supsi;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(ignoreUnknown=true)
public class Photo {


	//private int id;
	private String path;
	private String name;
	
	@JsonCreator
	public Photo(
			//@JsonProperty("id") int id,
			@JsonProperty("path") String path,
			@JsonProperty("name") String name
			)
	{
		//this.id = id;
		this.path = path;
		this.name = name;
	}

	/*@JsonProperty("id")
	public int getId() {
		return id;
	}*/


	@JsonProperty("path")
	public String getPath() {
		return path;
	}

	@JsonProperty("name")
	public String getName() {
		return name;
	}

	
	@Override
	public String toString() {
		return "Photo [name=" + name + ", path=" + path + "]";
	}

}
