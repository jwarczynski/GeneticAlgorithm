module com.example.graphvisualizer {
    requires javafx.controls;
    requires javafx.fxml;


    opens com.example.graphvisualizer to javafx.fxml;
    exports com.example.graphvisualizer;
}