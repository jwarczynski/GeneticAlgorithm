package com.example.graphvisualizer;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;

import java.io.BufferedReader;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class HelloController {
    private int active = 0;
    private int n = 50;
    private int dist = 350;
    private double X = 400;
    private double Y = 400;
    private List <Vortex> graph;
    private List <Edge> edges;
    private List<Color> colors;
    @FXML
    private Pane mainWindow;
    @FXML
    private TextField vortexChooser;
    @FXML
    private Button next;
    @FXML
    private Button prev;
    @FXML
    private Button clear;
    public void initialize() throws IOException, ClassNotFoundException, IllegalAccessException {
        //setColors();
        allColors();
        loadGraph("queen6.txt");
        colorGraph("gene_q.txt");
    }

    private void colorGraph(String filename) throws IOException {
        Path path;
        int i;
        BufferedReader br;
        String input;
        path = Paths.get(filename);
        br = Files.newBufferedReader(path);
        int maxColors = Integer.parseInt(br.readLine());
        int diff = colors.size()/maxColors-1;
        i = 0;
        try{
            while((input = br.readLine()) != null)
            {
                graph.get(i).setFilling(colors.get((Integer.parseInt(input)-1)*diff));
                i++;
            }
        }finally {
            if (br!=null)
            {
                br.close();
            }
        }
    }

    private void loadGraph(String filename) throws IOException {
        String input;
        Path path = Paths.get(filename);
        BufferedReader br= Files.newBufferedReader(path);
        input = br.readLine();
        n = Integer.parseInt(input);
        this.graph = new ArrayList<>();
        this.edges = new ArrayList<>();
        for(int i = 0; i < n; i++){
            graph.add(new Vortex(X, Y, dist, i, n));

        }
        int i = 0;
        try{
            while((input = br.readLine()) != null)
            {
                String[] itemPieces = input.split(" ");
                Vortex a = graph.get(Integer.parseInt(itemPieces[0])-1);
                Vortex b = graph.get(Integer.parseInt(itemPieces[1])-1);
                a.addAdjecent(b);
                b.addAdjecent(a);
                edges.add(new Edge(a.getX(), a.getY(), b.getX(), b.getY()));
                mainWindow.getChildren().add(edges.get(i).getSkin());
                i++;
            }
        }finally {
            if (br!=null)
            {
                br.close();
            }
        }

        for(Vortex v:graph){
            mainWindow.getChildren().add(v.getSkin());
            mainWindow.getChildren().add(v.getNum());
        }
    }

    private void allColors() throws ClassNotFoundException, IllegalAccessException {
        colors = new ArrayList<>();
        Class clazz = Class.forName("javafx.scene.paint.Color");
        if (clazz != null) {
            Field[] field = clazz.getFields();
            for (Field f : field) {
                Object obj = f.get(null);
                if (obj instanceof Color) {
                    colors.add((Color) obj);
                }

            }
        }
        colors.sort(new Comparator<Color>() {
            @Override
            public int compare(Color o1, Color o2) {
                return o1.toString().compareTo(o2.toString());
            }
        });
        colors.remove(Color.TRANSPARENT);
        colors.remove(Color.WHITE);
        colors.remove(Color.BLACK);

    }
    public void handleTextEdit(){
        active  = Integer.parseInt(vortexChooser.getText());
        changeShown(active);

    }

    private void changeShown(int i) {
        for(Vortex v: graph){
            v.hide();
        }
        for(Edge e: edges){
            e.getSkin().setOpacity(0);
        }
        if(i == 0){
            graphClear();
        }
        else if(i > 0 && i <= n){
            prev.setVisible(true);
            next.setVisible(true);
            graph.get(i -1).activate();
            for(Edge e: edges){
                e.activate(graph.get(i -1));
            }
            if(i == n){
                next.setVisible(false);
            }
        }
    }

    private void graphClear() {
        for(Vortex v: graph){
            v.deactivate();
        }
        for(Edge e: edges){
            e.deactivate();
        }
        prev.setVisible(false);
        next.setVisible(true);
    }

    public void handleNext(){
        active ++;
        vortexChooser.setText(String.valueOf(active));
        changeShown(active);
    }
    public void handlePrevious(){
        active--;
        vortexChooser.setText(String.valueOf(active));
        changeShown(active);
    }
    public void handleClear(){
        active = 0;
        vortexChooser.setText(String.valueOf(active));
        graphClear();
    }

    public void setColors() {
        colors = new ArrayList<>();
        colors.add(Color.AZURE);
        colors.add(Color.BLUE);
        colors.add(Color.YELLOW);
        colors.add(Color.RED);
        colors.add(Color.GREEN);
        colors.add(Color.GRAY);
    }
}