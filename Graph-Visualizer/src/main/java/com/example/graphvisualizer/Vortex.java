package com.example.graphvisualizer;

import javafx.fxml.FXML;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.text.Font;
import javafx.scene.text.Text;

import java.util.ArrayList;
import java.util.List;

public class Vortex {

    private boolean smallActive = false;
    private boolean bigActive = false;
    private int id;
    private double x;
    private double y;
    @FXML
    private Color filling;
    @FXML
    private Circle skin;
    @FXML
    private Text num;

    private List <Vortex> adjecent;

    public Vortex(double centerX, double centerY, double dist, int id, int n) {
        this.id = id;
        this.filling = Color.WHITE;
        this.skin = new Circle();
        double angle = 360.0/n*(id+1);
        this.x = centerX + dist*Math.sin(Math.toRadians(angle));
        this.y = centerY - dist*Math.cos(Math.toRadians(angle));
        skin.setCenterX(this.x);
        skin.setCenterY(this.y);
        skin.setRadius(20);
        skin.setFill(filling);
        skin.setStrokeWidth(2);
        skin.setStroke(Color.BLACK);
        this.num = new Text();
        num.setText(String.valueOf(id+1));
        num.setX(this.x);
        num.setY(this.y);
        num.setFont(Font.font("Arial"));
        adjecent = new ArrayList<>();
    }

    public void setFilling(Color filling) {
        this.filling = filling;
        this.skin.setFill(filling);
    }

    public boolean isSmallActive() {
        return smallActive;
    }

    public boolean isBigActive() {
        return bigActive;
    }

    public Circle getSkin() {
        return skin;
    }

    public double getX() {
        return x;
    }

    public double getY() {
        return y;
    }

    public boolean addAdjecent(Vortex adj){
       if(! adjecent.contains(adj)){
           adjecent.add(adj);
           return true;
       }
       else{
           return false;
       }
    }

    public List<Vortex> getAdjecent() {
        return adjecent;
    }

    public Text getNum() {
        return num;
    }

    public void activate(){
        skin.setStroke(Color.RED);
        skin.setOpacity(1);
        num.setVisible(true);
        for(Vortex v: adjecent){
            v.getSkin().setOpacity(1);
            v.getSkin().setStroke(Color.RED);
            v.getNum().setVisible(true);
        }
    }


    public void hide(){
        skin.setOpacity(0);
        num.setVisible(false);
    }
    public void deactivate(){
        skin.setStroke(Color.BLACK);
        skin.setOpacity(1);
        num.setVisible(true);
    }
}
